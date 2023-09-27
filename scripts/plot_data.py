import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "../data/state.txt"

columns = []
data = []

with open(file_path, 'r') as file:
    in_data_section = False
    in_columns_section = False

    for line in file:
        line = line.strip()

        if line == 'data':
            in_data_section = True
            continue
        if line.startswith('columns'):
            in_columns_section = True
            continue

        if in_data_section:
            values = line.split(',')
            data.append(values)
        elif in_columns_section:
            in_columns_section = False
            columns = line.split(',')[:-1]

df = pd.DataFrame(data, columns=columns)

df = df.apply(pd.to_numeric, errors='ignore')

altitude = df['altitude'].to_numpy(dtype=np.float64)
velocity = df['velocity'].to_numpy(dtype=np.float64)

plt.plot(altitude)
plt.show()