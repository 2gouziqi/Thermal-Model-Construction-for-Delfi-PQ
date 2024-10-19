import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from data_loader import get_group

# Define get_fit_data inside exponential_fit.py
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

# Define the piecewise exponential function
def piecewise_Exp(x, a, b, c, d, fe_mean):
    return np.piecewise(
        x, 
        [x < 360 * (1 - fe_mean), x >= 360 * (1 - fe_mean)],
        [lambda x: a * (1 - np.exp(-x * b)) + c,
         lambda x: d * (np.exp(-x * b)) + c]
    )

def main():
    # Define file paths
    tle_file = os.path.join('data', 'tles', 'Delfi-PQ_TLEs_2022-03-16.txt')
    temperature_file = os.path.join('data', 'temperature_file', 'temperatures_EU1XX_JA0CAW_PY4ZBZ_filtered.csv')

    # Load the data
    thetas, dist_sun_sat, groups, beta_angles, fe, heights = get_group(tle_file, temperature_file)

    # Get the required data using the get_fit_data function
    theta, temp, beta, fe = get_fit_data('PanelYpTemperature', thetas, groups, beta_angles, fe)

    # Find the mean of fe
    fe_mean = np.mean(fe)
    tran = 180 * (1 + fe_mean)  # Transition angle where behavior changes

    # Split the data into two regions
    theta1, theta2, temp1, temp2 = [], [], [], []
    for i in range(len(theta)):
        if theta[i] <= tran:
            theta1.append(theta[i])
            temp1.append(temp[i])
        else:
            theta2.append(theta[i])
            temp2.append(temp[i])

    # Rearrange theta values for better fitting
    new_th = np.hstack((theta2 - 360 * np.ones(len(theta2)), theta1))
    new_th = new_th + (360 - tran) * np.ones(len(new_th))
    new_temp = np.hstack((temp2, temp1))

    # Plot the rearranged data
    plt.scatter(new_th, new_temp)
    plt.ylabel('Temperature [C]', fontsize=14)
    plt.xlabel('New theta [deg]', fontsize=14)
    plt.legend(['Data'], fontsize=12)
    plt.show()

    # Fit the data using curve fitting
    params_exp, _ = optimize.curve_fit(lambda x, a, b, c, d: piecewise_Exp(x, a, b, c, d, fe_mean), 
                                       new_th, new_temp, maxfev=500)
    print(f"Fitted parameters: {params_exp}")

    # Plot the fitted curve along with the data points
    xp = np.linspace(0, 360, 1000)
    plt.scatter(new_th, new_temp, label='Data')
    plt.plot(xp, piecewise_Exp(xp, *params_exp, fe_mean), label='Exponential Fit', color='red')
    plt.xlabel('New theta [deg]', fontsize=14)
    plt.ylabel('Temperature [C]', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Temperature vs. Theta with Exponential Fitting', fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
