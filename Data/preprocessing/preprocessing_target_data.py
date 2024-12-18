import os
import glob
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# filename_user = glob.glob('Data/Target/HS/HS_TrainData.csv')
# label_file = glob.glob('Data/Target/HS/HS_TrainLabel.csv')
# filename_user = glob.glob('Data/Target/HS/SensorData.csv')
# label_file = glob.glob('Data/Target/HS/SensorLabel.csv')
# save_path = 'Data/Target/HS/'

filename_user = glob.glob('Data/Target/CL/cl_TrainData_?.csv')
label_file = glob.glob('Data/Target/CL/cl_TrainLabel_?.csv')
save_path = './Data/Target/CHECK/'

target_list = ['Target1', 'Target2', 'Target3', 'Target4']

in_ch = 6
segment_size = 250
seed = 0
random.seed(seed)
np.random.seed(seed)

sensor_data = []
target_data = []

# for fn in filename:
#     data_list = pd.read_csv(fn, header=None)
#     label = os.path.split(fn)[-1].split('_')[0]
#     target_name = target_list.index(label)
    
#     sensor_data.append(data_list.values)
#     target_data.append(target_name)

file_idx = 0
for fn in filename_user:
    data_list_user = pd.read_csv(fn, header=None)
    lable_list = pd.read_csv(label_file[file_idx], header=None)
    file_idx += 1
    
    for idx in range(len(data_list_user)):
        sensor_temp = []
        label_idx = lable_list.values[idx][0]
        target_data.append(label_idx)

        for n in range(0, in_ch*segment_size, in_ch):
            sensor_temp.append([data_list_user.values[idx][n], data_list_user.values[idx][n+1], data_list_user.values[idx][n+2], data_list_user.values[idx][n+3], data_list_user.values[idx][n+4], data_list_user.values[idx][n+5]])

        sensor_data.append(sensor_temp)

# max_acc = 0.5; min_acc = -0.5
# max_gyro = 0.1; min_gyro = -0.1

# for idx in range(60):
#     none_data = []
#     for s in range(250):
#         temp_none = []
#         for c in range(6):
#             if c < 3:
#                 temp_none.append((random.random() * (max_acc - min_acc)) + min_acc)
#             else:
#                 temp_none.append((random.random() * (max_gyro - min_gyro)) + min_gyro)
#         none_data.append([temp_none[0], temp_none[1], temp_none[2], temp_none[3], temp_none[4], temp_none[5]])
#     sensor_data.append(none_data)
#     target_data.append(9)

print('Load Data')
sensor_data = np.array(sensor_data)     # (n, 250, 6)
target_data = np.array(target_data)

print(sensor_data.shape)
print(target_data.shape)

sensor_data = sensor_data.reshape(-1, in_ch*segment_size)

df_data = pd.DataFrame(sensor_data)
df_data.to_csv(save_path + "SensorData.csv", index=False, header=False)
df_target = pd.DataFrame(target_data)
df_target.to_csv(save_path + "SensorLabel.csv", index=False, header=False)

# # Normalization Acc
# # min_acc = -78; max_acc = 78
# min_acc = np.min(sensor_data[:, :, :3]); max_acc = np.max(sensor_data[:, :, :3])
# sensor_data[:, :, :3] = (sensor_data[:, :, :3] - min_acc) / (max_acc - min_acc)
# # Normalization Gyro
# # min_gyro = -1000; max_gyro = 1000
# min_gyro = np.min(sensor_data[:, :, 3:]); max_gyro = np.max(sensor_data[:, :, 3:])
# sensor_data[:, :, 3:] = (sensor_data[:, :, 3:] - min_gyro) / (max_gyro - min_gyro)

train_x, test_x, train_y, test_y = train_test_split(sensor_data, target_data, test_size=0.2, random_state=seed, shuffle=True, stratify=target_data)

train_x = np.array(train_x).astype(np.float32)
test_x = np.array(test_x).astype(np.float32)

train_x = train_x.reshape(-1, 6, segment_size)
test_x = test_x.reshape(-1, 6, segment_size)

np.save(save_path + 'train_x.npy', train_x.astype(np.float32))
np.save(save_path + 'train_y.npy', train_y)
np.save(save_path + 'test_x.npy', test_x.astype(np.float32))
np.save(save_path + 'test_y.npy', test_y)

print('save')

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)