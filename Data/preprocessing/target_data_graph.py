import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import MinMaxScaler

filename = glob.glob('Data/Target/test/*.csv')
save_path = 'Data/Target/test/png/'

for fn in filename:
    data_list = pd.read_csv(fn, header=None)
    print(os.path.basename(fn))

    sensor_data = data_list.values
    acc_x = []; acc_y = []; acc_z = []
    gyro_x = []; gyro_y = []; gyro_z = []

    for s in sensor_data:
        acc_x.append(s[0]); acc_y.append(s[1]); acc_z.append(s[2])
        gyro_x.append(s[3]); gyro_y.append(s[4]); gyro_z.append(s[5])

    xs = np.linspace(0, 2.5, len(sensor_data))

    fig, (acc, gyro) = plt.subplots(2)

    acc.set_title('Accelerometer')
    acc.plot(xs, acc_x, 'r', label='x')
    acc.plot(xs, acc_y, 'g', label='y')
    acc.plot(xs, acc_z, 'b', label='z')
    acc.set_xlabel('Time')
    acc.set_ylabel('m/s^2')
    acc.legend(loc='upper right')
    
    gyro.set_title('Gyroscope')
    gyro.plot(xs, gyro_x, 'r', label='x')
    gyro.plot(xs, gyro_y, 'g', label='y')
    gyro.plot(xs, gyro_z, 'b', label='z')
    gyro.set_xlabel('Time')
    gyro.set_ylabel('˚/s')
    gyro.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(save_path + os.path.splitext(os.path.split(fn)[-1])[0])