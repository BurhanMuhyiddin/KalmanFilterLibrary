from helper_functions import read_data, plot_data

import numpy as np
import matplotlib.pyplot as plt

import json

SCENARIO1_PARAMS = "scenarios/scenario1.json"
PATH_PLOT_PARAMS = "parameters/plot_params.json"
PATH_GT_ALTITUDE = "data/LKF/gtAltitude.txt"
PATH_MEAS_ALTITUDE = "data/LKF/measAltitude.txt"
PATH_GT_VELOCITY = "data/LKF/gtVelocity.txt"
PATH_ESTIMATED_STATE = "data/LKF/state.txt"
PATH_ESTIMATE_COVARIANCE = "data/LKF/estimateCovariance.txt"

# Read plot parameters
plot_params_file = open(PATH_PLOT_PARAMS)

plot_params = json.load(plot_params_file)

plot_gt_data = plot_params["flags"]["plot_gt_data"]
plot_out_data = plot_params["flags"]["plot_out_data"]
plot_meas_data = plot_params["flags"]["plot_meas_data"]
plot_conf_interval = plot_params["flags"]["plot_conf_interval"]

# Read time sequence for plots
scenario1_params_file = open(SCENARIO1_PARAMS)
scenario1_params = json.load(scenario1_params_file)
dt = scenario1_params["KF"]["dt"]
N = scenario1_params["scenario"]["N"]
t = np.arange(0, N*dt, dt)

# Read data
gt_alt, column_names_alt = read_data(file_path = PATH_GT_ALTITUDE)
gt_vel, column_names_vel = read_data(file_path = PATH_GT_VELOCITY)
gt_state = np.vstack([gt_alt, gt_vel])
gt_state = np.transpose(gt_state, (1, 0, 2))

meas_alt, column_names_alt = read_data(file_path = PATH_MEAS_ALTITUDE)
meas_alt = np.transpose(meas_alt, (1, 0, 2))

estimated_states, column_names = read_data(file_path = PATH_ESTIMATED_STATE)
estimate_covariance, _ = read_data(file_path = PATH_ESTIMATE_COVARIANCE)

# Create data dictionary to pass to the plot function
data_dict = {
    "plot_gt": plot_gt_data,
    "plot_out": plot_out_data,
    "plot_meas": plot_meas_data,
    "plot_conf_int": plot_conf_interval,

    "gt_data": gt_state,
    "est_data": estimated_states,
    "est_covariance": estimate_covariance,
    "meas_data": meas_alt
}

# Plot data with the specified parameters
plot_data(data_dict, t, columns=column_names)