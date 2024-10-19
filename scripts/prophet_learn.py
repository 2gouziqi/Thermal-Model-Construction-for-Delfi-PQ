import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta
from data_loader import get_group
from prophet import Prophet
from prophet.diagnostics import performance_metrics, cross_validation
from prophet.serialize import model_to_json, model_from_json

# Define the get_fit_data function
def get_fit_data(subsystem, thetas, groups, beta_angles, fe):
    x3_values = []
    y3_values = []
    for key, item in thetas.items():
        for temp_key, theta in item.items():
            tmps = groups[key][1]
            for t in tmps:
                if t['timestamp'] == temp_key:
                    x3_values.append(np.rad2deg(theta))
                    y3_values.append(float(t[subsystem]))
    
    x5_values = []
    y5_values = []
    for key, item in beta_angles.items():
        x5_values.append(np.rad2deg(item))
        y5_values.append(fe[key])
    
    return x3_values, y3_values, x5_values, y5_values

def main():
    require_training = True  # Set to False to load the model directly

    print("Processing data...")

    # Define file paths
    tle_file = os.path.join('data', 'tles', 'Delfi-PQ_TLEs_2022-03-16.txt')
    temperature_file = os.path.join('data', 'temperature_file', 'temperatures_EU1XX_JA0CAW_PY4ZBZ_filtered.csv')

    # Load data
    thetas, dist_sun_sat, groups, beta_angles, fe, heights = get_group(tle_file, temperature_file)
    theta, temp, beta, fe = get_fit_data('PanelYpTemperature', thetas, groups, beta_angles, fe)

    fe_mean = np.mean(fe)
    tran = 180 * (1 + fe_mean)  # Transition angle

    theta1, theta2, temp1, temp2 = [], [], [], []
    for i in range(len(theta)):
        if theta[i] <= tran:
            theta1.append(theta[i])
            temp1.append(temp[i])
        else:
            theta2.append(theta[i])
            temp2.append(temp[i])

    # Calculate its period (LEO)
    avg_height = np.mean([h.km for h in heights.values()])
    velocity = np.sqrt(198600.5 / (6378.14 + avg_height))
    period = 2 * np.pi * (6378.14 + avg_height) / velocity
    t_per_degree = period / 360.0

    # Set t0 for the calculation (arbitrarily)
    t0 = datetime(2022, 5, 1, 0, 0, 0)

    new_th = np.hstack((theta2 - 360 * np.ones(len(theta2)), theta1))
    new_th = new_th + (360 - tran) * np.ones(len(new_th))

    new_time = [t0 + timedelta(seconds=theta * t_per_degree) for theta in new_th]
    new_temp = np.hstack((temp2, temp1))

    # Remove outlier(s)
    for i in range(len(new_time) - 1):
        min_t = t0 + timedelta(seconds=250 * t_per_degree)
        max_t = t0 + timedelta(seconds=270 * t_per_degree)
        if min_t < new_time[i] < max_t:
            new_time = np.delete(new_time, i)
            new_temp = np.delete(new_temp, i)

    d = {'ds': new_time, 'y': new_temp}
    df = pd.DataFrame(d)

    changepoints_min, changepoints_max = min(new_time), max(new_time)
    changepoints = []
    i = 0
    while changepoints_min + timedelta(seconds=240 * t_per_degree + i * period) < changepoints_max:
        changepoints.append(changepoints_min + timedelta(seconds=240 * t_per_degree + i * period))
        i += 1

    print("Completed")

    # Use Prophet to train and predict
    if require_training:
        m = Prophet(changepoints=changepoints, changepoint_prior_scale=0.0001)
        m.add_seasonality(name='minutely', period=period / 86400.0, fourier_order=2)
        m.fit(df)

        # Save the trained model
        with open('serialized_model.json', 'w') as fout:
            json.dump(model_to_json(m), fout)
    else:
        # Load the saved model
        with open('serialized_model.json', 'r') as fin:
            m = model_from_json(json.load(fin))

    # Visualize the prediction
    future = pd.DataFrame({'ds': [t0 + timedelta(seconds=i * t_per_degree) for i in range(360)]})
    forecast = m.predict(future)
    figure = m.plot(forecast)

    # Calculate r2 and RMSE
    yhat = m.predict(df)
    metric_df = yhat.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
    metric_df.dropna(inplace=True)

    r2 = r2_score(metric_df.y, metric_df.yhat)
    rmse = np.sqrt(mean_squared_error(metric_df.y, metric_df.yhat))
    print(f"Prophet:\nr2score: {r2}\nrmse: {rmse}")

    # Plot the Prophet prediction ± 3*σ
    plt.plot(forecast['ds'], forecast['yhat'] + 3 * rmse, color='blue', ls='--')
    plt.plot(forecast['ds'], forecast['yhat'] - 3 * rmse, color='blue', ls='--')
    plt.legend(['Data points', 'Prophet prediction', 'Prophet prediction ± 3σ'])

    # Save the figure
    figure.savefig('output_prophet.png')

    plt.show()

if __name__ == "__main__":
    main()
