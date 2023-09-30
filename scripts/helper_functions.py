import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def read_data(file_path: str):
    if not os.path.exists(file_path):
        print(f"File {file_path} doesn't exist!!!")
        return
    
    columns = []
    data = []
    n_rows = 0
    n_cols = 0
    
    with open(file_path, 'r') as file:
        in_data_section = False
        in_columns_section = False
        in_dimension_section = False

        for line in file:
            line = line.strip()

            if line == 'data':
                in_data_section = True
                continue
            if line == 'columns':
                in_columns_section = True
                continue
            if line == 'dimension':
                in_dimension_section = True
                continue

            if in_data_section:
                values = line.split(',')
                data.append(values)
            elif in_columns_section:
                in_columns_section = False
                columns = line.split(',')[:-1]
            elif in_dimension_section:
                in_dimension_section = False
                matrix_size = line.split(',')
                n_rows = int(matrix_size[0])
                n_cols = int(matrix_size[1])

    # if len(columns) != 0:
    #     df = pd.DataFrame(data, columns=columns)
    # else:
    #     df = pd.DataFrame(data)

    # df = df.apply(pd.to_numeric, errors='ignore')

    data = np.array(data, dtype=np.float64)
    data = data.reshape((-1, n_rows, n_cols))

    return data, columns


def plot_data(data_dict: dict, t: np.array, columns: list):
    plot_gt = data_dict["plot_gt"]
    plot_out = data_dict["plot_out"]
    plot_meas = data_dict["plot_meas"]
    plot_conf_int = data_dict["plot_conf_int"]

    gt_data = data_dict["gt_data"]
    est_data = data_dict["est_data"]
    est_covariance = data_dict["est_covariance"]
    meas_data = data_dict["meas_data"]

    for i, column in enumerate(columns):
        legends = []

        current_gt_state = gt_data[:, i, 0]
        current_state = est_data[:, i, 0]
        current_state_viariance = est_covariance[:, i, i]

        meas_available = False
        if i < meas_data.shape[1]:
            meas_available = True
            current_state_meas = meas_data[:, i, 0]

        plt.figure()

        if plot_out:
            legends.append(f"est: {column}")
            plt.plot(t, current_state)
        if plot_gt:
            legends.append(f"gt: {column}")
            plt.plot(t, current_gt_state)
        if plot_meas and meas_available:
            legends.append(f"meas: {column}")
            plt.plot(t, current_state_meas)

        if plot_conf_int:
            plot_confidence_interval(current_state, current_state_viariance, t)

        plt.xlabel('time')
        plt.ylabel(f'{column}')
        plt.title(f'{column} Plot')
        plt.legend(legends)

    plt.show()

def plot_confidence_interval(x: np.array, estimate_covariance: np.array, t: np.array, pct_conf: int=95):
        xStd = np.sqrt(estimate_covariance)
        zScore = norm.ppf(1-(1-pct_conf/100)/2)
        xConf = xStd * zScore

        art = plt.fill_between(t, x-xConf, x+xConf, color='#FFFF99', edgecolor='#FF804D',
                                linewidth=1, alpha=0.6, label='95% confidence interval')
        art.set_edgecolor('#FF804D')