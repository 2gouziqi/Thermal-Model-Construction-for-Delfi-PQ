from skyfield.api import load, utc
from skyfield.framelib import ecliptic_frame
from skyfield.toposlib import wgs84
import csv
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Load time scale and celestial data
ts = load.timescale()
eph = load('de421.bsp')

# Get the epoch UTC time
def get_time(epochs):
    return epochs[0].epoch.utc

# Calculate the ecliptic true solar longitude (0-360°) for a given epoch
def lon_angle(epoch):
    t = epoch.epoch
    astrometric = eph['Earth'].at(t).observe(eph['Sun'])
    lat, lon, distance = astrometric.apparent().frame_latlon(ecliptic_frame)
    return lon.degrees

# Calculate beta angle for a given epoch
def beta_angle(epoch):
    lon = np.deg2rad(lon_angle(epoch))  # Ecliptic true solar longitude
    omega = epoch.model.nodeo           # RAAN
    i = epoch.model.inclo               # Inclination
    eps = np.deg2rad(23.45)             # Obliquity of the ecliptic (tilt of Earth's axis)
    
    # Beta angle formula
    return np.arcsin(
        np.cos(lon) * np.sin(omega) * np.sin(i) -
        np.sin(lon) * np.cos(eps) * np.cos(omega) * np.sin(i) +
        np.sin(lon) * np.sin(eps) * np.cos(i)
    )

# Calculate angle between two vectors
def angle_between_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)

# Calculate the theta angle
def theta(u_vector, st_vector, norm_vect):
    angle = angle_between_vectors(u_vector, st_vector)
    cross_product_sign = np.dot(np.cross(u_vector, st_vector), norm_vect)
    
    # Adjust the angle range from 0-pi to 0-2pi
    if cross_product_sign < 0:
        angle = 2 * np.pi - angle

    return angle


def get_group(tle, tempfile):
    if not os.path.exists(tle):
        raise FileNotFoundError(f"TLE file not found: {tle}")
    
    if not os.path.exists(tempfile):
        raise FileNotFoundError(f"Temperature file not found: {tempfile}")

    # Load TLEs as epochs and group them by time
    epochs = load.tle_file(tle)
    unique_epochs = set([e.epoch for e in epochs])
    grouped_epochs = [[e for e in epochs if e.epoch == ue] for ue in unique_epochs]
    
    if len(grouped_epochs) == 0:
        raise ValueError("No valid epochs found in the TLE file.")
    
    grouped_epochs.sort(key=get_time)
    
    # Load temperature data from CSV file
    with open(tempfile, encoding='utf-8-sig') as csv_file:
        temperatures = list(csv.DictReader(csv_file, delimiter=','))
    
    # Parse timestamps and filter out temperatures beyond the final epoch time
    final_epoch_time = grouped_epochs[-1][0].epoch.utc_datetime()
    temperatures = [
        {**t, 'timestamp': datetime.strptime(t['timestamp'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=utc)}
        for t in temperatures
    ]
    temperatures = [t for t in temperatures if t['timestamp'] <= final_epoch_time]

    # Group epochs with their corresponding temperatures
    groups = {}
    for i, epoch_group in enumerate(grouped_epochs[:-1]):
        temp_list = [
            t for t in temperatures 
            if epoch_group[0].epoch.utc_datetime() < t['timestamp'] < grouped_epochs[i+1][0].epoch.utc_datetime()
        ]
        groups[epoch_group[0].epoch] = (epoch_group, temp_list)

    groups = {key: value for key, value in groups.items() if value[1]}

    # Get solar vectors (Earth-Sun vector)
    solar_vectors = {key: eph['Earth'].at(key).observe(eph['Sun']).apparent().position.km for key in groups.keys()}
    beta_angles = {key: beta_angle(item[0][0]) for key, item in groups.items()}
    r_vectors = {key: (item[0][0].at(key), item[0][0].at(key + timedelta(hours=0.1))) for key, item in groups.items()}
    norm_vectors_surface = {key: np.cross(r0.position.km, r1.position.km) for key, (r0, r1) in r_vectors.items()}
    
    # Scale calculation for projection vector
    a_values = {key: np.linalg.norm(np.sin(beta_angles[key]) * np.linalg.norm(solar_vectors[key]) /
                np.linalg.norm(norm_vect)) * (-1 if angle_between_vectors(norm_vect, solar_vectors[key]) > 0.5 * np.pi else 1)
                for key, norm_vect in norm_vectors_surface.items()}
    u_vectors = {key: solar_vectors[key] - a_values[key] * norm_vectors_surface[key] for key in norm_vectors_surface.keys()}
    
    st_vectors = {key: {temp['timestamp']: item[0][0].at(ts.from_datetime(temp['timestamp'])) for temp in item[1]} for key, item in groups.items()}
    thetas = {key: {temp_time: theta(u_vectors[key], st.position.km, norm_vectors_surface[key]) for temp_time, st in temp_vectors.items()}
              for key, temp_vectors in st_vectors.items()}
    dist_sun_sat = {key: {temp_time: np.linalg.norm(st.position.km - solar_vectors[key]) for temp_time, st in temp_vectors.items()}
                    for key, temp_vectors in st_vectors.items()}
    
    h = {key: wgs84.height_of(item[0][0].at(key)) for key, item in groups.items()}
    Re = 6371000  # Earth's radius in meters
    beta_star = {key: np.arcsin(Re / (Re + h_val.m)) for key, h_val in h.items()}
    fe_values = {key: (np.arccos(np.sqrt(h_val.m**2 + 2 * Re * h_val.m) / ((Re + h_val.m) * np.cos(beta_angles[key]))) / np.pi)
                 if abs(beta_angles[key]) < beta_star[key] else 0 for key, h_val in h.items()}

    return thetas, dist_sun_sat, groups, beta_angles, fe_values, h

def plot(thetas, dist_sun_sat, groups, beta_angles, fe_values):
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    def scatter_plot(x_values, y_values, xlabel, ylabel, label=None, fig_name=None):
        plt.scatter(x_values, y_values, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if label:
            plt.legend()
        if fig_name:
            plt.savefig(os.path.join('plots', fig_name))  # Save in 'plots' folder
        plt.close()

    # Detect temperature-related columns (those with 'temp' or 'Temp')
    temp_columns = [col for group in groups.values() for col in group[1][0].keys() if 'temp' in col.lower()]
    
    if not temp_columns:
        print("No columns with 'temp' or 'Temp' found in the CSV file.")
        return

    # Plot theta angle vs. each detected temperature column
    for subsystem in temp_columns:
        x_theta, y_temp = [], []
        for key, theta_values in thetas.items():
            for temp_time, theta in theta_values.items():
                for temp in groups[key][1]:
                    if temp['timestamp'] == temp_time:
                        x_theta.append(np.rad2deg(theta))
                        y_temp.append(float(temp[subsystem]))
        
        scatter_plot(x_theta, y_temp, 'Theta Angle [deg]', f'Temperature [{subsystem}]', fig_name=f'plot_theta_vs_{subsystem}.png')

    # Plot theta angle vs. sun-satellite distance
    x_theta, y_distance = zip(*[(np.rad2deg(theta), dist_sun_sat[key][temp_time])
                                for key, theta_values in thetas.items()
                                for temp_time, theta in theta_values.items()])
    
    scatter_plot(x_theta, y_distance, 'Theta Angle [deg]', 'Sun-satellite Distance [km]', fig_name='plot_theta_vs_sun_distance.png')

    # Plot datetime vs. theta angle
    x_time, y_theta = zip(*[(temp_time, np.rad2deg(theta))
                            for key, theta_values in thetas.items()
                            for temp_time, theta in theta_values.items()])
    
    # Plot datetime vs. theta angle as scatter plot
    plt.scatter(matplotlib.dates.date2num(x_time), y_theta)
    plt.xlabel('Satellite Time')
    plt.ylabel('Theta Angle [deg]')
    plt.savefig(os.path.join('plots', 'plot_datetime_vs_theta.png'))  # Save in 'plots' folder
    plt.close()

    # Plot beta angle vs. time as scatter plot
    x_time = [item[0][0].epoch.utc_datetime() for key, item in groups.items()]
    y_beta = [beta_angle(item[0][0]) * 180 / np.pi for key, item in groups.items()]
    
    plt.scatter(matplotlib.dates.date2num(x_time), y_beta)
    plt.xlabel('Epoch Time')
    plt.ylabel('Beta Angle [deg]')
    plt.savefig(os.path.join('plots', 'plot_beta_vs_time.png'))  # Save in 'plots' folder
    plt.close()

def main():
    tle_file = 'data\\tles\\Delfi-PQ_TLEs_2022-03-16.txt'
    temperature_file = 'data\\temperature_file\\temperatures_EU1XX_JA0CAW_PY4ZBZ_filtered.csv'
    
    thetas, dist_sun_sat, groups, beta_angles, fe_values, altitudes = get_group(tle_file, temperature_file)    
    plot(thetas, dist_sun_sat, groups, beta_angles, fe_values)

if __name__ == "__main__":
    main()
