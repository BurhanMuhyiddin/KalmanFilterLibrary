from helper_functions import read_data, plot_data

import numpy as np
import matplotlib.pyplot as plt

import json

# SCENARIO1_PARAMS = "scenarios/scenario1.json"
# PATH_PLOT_PARAMS = "parameters/plot_params.json"
# PATH_GT_ALTITUDE = "data/gtAltitude.txt"
# PATH_MEAS_ALTITUDE = "data/measAltitude.txt"
# PATH_GT_VELOCITY = "data/gtVelocity.txt"
# PATH_ESTIMATED_STATE = "data/state.txt"
# PATH_ESTIMATE_COVARIANCE = "data/estimateCovariance.txt"

SCENARIO2_PARAMS = "scenarios/scenario2.json"
PATH_PLOT_PARAMS = "parameters/plot_params.json"
PATH_GT_STATE = "data/UKF/gtState.txt"
PATH_ESTIMATED_STATE = "data/UKF/state.txt"
PATH_ESTIMATE_COVARIANCE = "data/UKF/estimateCovariance.txt"

# Read plot parameters
plot_params_file = open(PATH_PLOT_PARAMS)

plot_params = json.load(plot_params_file)

plot_gt_data = plot_params["flags"]["plot_gt_data"]
plot_out_data = plot_params["flags"]["plot_out_data"]
plot_meas_data = plot_params["flags"]["plot_meas_data"]
plot_conf_interval = plot_params["flags"]["plot_conf_interval"]

# Read data
gt_state, column_names_gt_state = read_data(file_path = PATH_GT_STATE)

estimated_states, column_names = read_data(file_path = PATH_ESTIMATED_STATE)
estimate_covariance, _ = read_data(file_path = PATH_ESTIMATE_COVARIANCE)

# Read time sequence for plots
scenario2_params_file = open(SCENARIO2_PARAMS)
scenario2_params = json.load(scenario2_params_file)
dt = scenario2_params["KF"]["dt"]
# N = scenario2_params["scenario"]["N"]
N = gt_state.shape[0]
t = np.arange(0, N*dt, dt)

# Create data dictionary to pass to the plot function
data_dict = {
    "plot_gt": plot_gt_data,
    "plot_out": plot_out_data,
    "plot_meas": plot_meas_data,
    "plot_conf_int": plot_conf_interval,

    "gt_data": gt_state,
    "est_data": estimated_states,
    "est_covariance": estimate_covariance,
    "meas_data": np.empty_like(gt_state)
}

# Plot data with the specified parameters
plot_data(data_dict, t, columns=column_names)

gt_x = gt_state[:, 0, 0]
gt_y = gt_state[:, 3, 0]

x = estimated_states[:, 0, 0]
y = estimated_states[:, 3, 0]

plt.figure()

plt.plot(x, y)
plt.plot(gt_x, gt_y)
plt.legend(["est state", "gt state"])
plt.title("Trajectory Plot")

plt.show()