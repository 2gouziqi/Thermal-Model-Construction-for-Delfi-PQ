import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from data_loader import get_group

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

# Load and process data
def main():
    print("Processing data...")

    # Define file paths
    tle_file = os.path.join('data', 'tles', 'Delfi-PQ_TLEs_2022-03-16.txt')
    temperature_file = os.path.join('data', 'temperature_file', 'temperatures_EU1XX_JA0CAW_PY4ZBZ_filtered.csv')

    # Load data using get_group and get_fit_data
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

    new_th = np.hstack((theta2 - 360 * np.ones(len(theta2)), theta1))
    new_th = new_th + (360 - tran) * np.ones(len(new_th))
    new_temp = np.hstack((temp2, temp1))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(new_th), np.asarray(new_temp), test_size=0.1, random_state=42
    )

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Use SVR with RBF kernel to learn and predict
    regr = SVR(kernel='rbf')
    regr.fit(X_train, y_train)

    preds = regr.predict(X_test)

    r2 = r2_score(y_test, preds)
    print(f"r2score: {r2}")

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"rmse(sigma) = {rmse}")

    # Visualisation
    X_train_pred = np.linspace(0, 360, 1000).reshape(-1, 1)
    y_train_pred = regr.predict(X_train_pred)

    plt.figure(figsize=(12, 6))

    # Scatter plot for training and testing data
    plt.scatter(X_train, y_train, color='blue', label='Train Data')
    plt.scatter(X_test, y_test, color='red', label='Test Data')

    # Model prediction plot
    plt.plot(X_train_pred, y_train_pred, color='black', linewidth=3, label='Model')

    # Model ±3*σ prediction range
    plt.plot(X_train_pred, y_train_pred - 3 * rmse, color='blue', linewidth=2, label='Model ±3σ', ls='--')
    plt.plot(X_train_pred, y_train_pred + 3 * rmse, color='blue', linewidth=2, ls='--')

    # Add labels, title, and legend
    plt.xlabel('Theta Angle [deg]', fontsize=14)
    plt.ylabel('Temperature [C]', fontsize=14)
    plt.title('Support Vector Regression Model with RBF Kernel', fontsize=16)
    plt.legend(fontsize=12)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
