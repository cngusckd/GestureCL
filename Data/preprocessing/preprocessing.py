import os
import glob
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# filename_user = glob.glob('Data/Target/HS/HS_TrainData.csv')
# label_file = glob.glob('Data/Target/HS/HS_TrainLabel.csv')
filename = glob.glob('Data/Target/CL/cl_TrainData_?.csv')
label_file = glob.glob('Data/Target/CL/cl_TrainLabel_?.csv')
save_path = './Data/Target/CHECK/'

target_num = [0, 1, 2, 3]
target_list = ['Target1', 'Target2', 'Target3', 'Target4']

in_ch = 6
segment_size = 250
seed = 0
random.seed(seed)
np.random.seed(seed)

sensor_data = []
target_data = []
print(filename)
data_list_user = pd.read_csv(filename[3], header=None)
lable_list = pd.read_csv(label_file[3], header=None)



for idx in range(len(lable_list)):
    if lable_list.values[idx][0] in target_num:
        sensor_temp = []

        target_data.append(lable_list.values[idx][0])

        for n in range(0, in_ch*segment_size, in_ch):
            sensor_temp.append([data_list_user.values[idx][n], data_list_user.values[idx][n+1], data_list_user.values[idx][n+2], data_list_user.values[idx][n+3], data_list_user.values[idx][n+4], data_list_user.values[idx][n+5]])

        sensor_data.append(sensor_temp)


print('Load Data')
sensor_data = np.array(sensor_data)     # (n, 250, 6)
target_data = np.array(target_data)


print("Inference")
train_x, _, train_y, _ = train_test_split(sensor_data, target_data, test_size=0.2, random_state=seed, shuffle=True, stratify=target_data)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=seed, shuffle=True, stratify=train_y)
train_x = np.array(train_x).astype(np.float32)
val_x = np.array(val_x).astype(np.float32)

train_x = train_x.reshape(-1, 6, segment_size)
val_x = val_x.reshape(-1, 6, segment_size)

np.save(save_path + 'Inference/train_x.npy', train_x.astype(np.float32))
np.save(save_path + 'Inference/train_y.npy', train_y)
np.save(save_path + 'Inference/val_x.npy', val_x.astype(np.float32))
np.save(save_path + 'Inference/val_y.npy', val_y)


print("TL & CL")
# 5:5
Train_data, TL_data, Train_label, TL_label = train_test_split(sensor_data, target_data, test_size=0.5, random_state=seed, shuffle=True, stratify=target_data)

train_x, val_x, train_y, val_y = train_test_split(Train_data, Train_label, test_size=0.2, random_state=seed, shuffle=True, stratify=Train_label)
train_x = np.array(train_x).astype(np.float32)
val_x = np.array(val_x).astype(np.float32)

train_x = train_x.reshape(-1, 6, segment_size)
val_x = val_x.reshape(-1, 6, segment_size)

np.save(save_path + 'TL/train_x.npy', train_x.astype(np.float32))
np.save(save_path + 'TL/train_y.npy', train_y)
np.save(save_path + 'TL/val_x.npy', val_x.astype(np.float32))
np.save(save_path + 'TL/val_y.npy', val_y)

np.save(save_path + 'CL/train_x.npy', train_x.astype(np.float32))
np.save(save_path + 'CL/train_y.npy', train_y)
np.save(save_path + 'CL/val_x.npy', val_x.astype(np.float32))
np.save(save_path + 'CL/val_y.npy', val_y)

tl_x, test_x, tl_y, test_y = train_test_split(TL_data, TL_label, test_size=0.8, random_state=seed, shuffle=True, stratify=TL_label)
tl_x = np.array(tl_x).astype(np.float32)
test_x = np.array(test_x).astype(np.float32)

tl_x = tl_x.reshape(-1, in_ch*segment_size)
df_data = pd.DataFrame(tl_x)
df_data.to_csv(save_path + "TL/tl_TainData.csv", index=False, header=False)
df_target = pd.DataFrame(tl_y)
df_target.to_csv(save_path + "TL/tl_TainLabel.csv", index=False, header=False)

test_x = test_x.reshape(-1, in_ch*segment_size)
df_data = pd.DataFrame(test_x)
df_data.to_csv(save_path + "Inference/TestData.csv", index=False, header=False)
df_data.to_csv(save_path + "TL/TestData.csv", index=False, header=False)
df_data.to_csv(save_path + "CL/TestData.csv", index=False, header=False)
df_target = pd.DataFrame(test_y)
df_target.to_csv(save_path + "Inference/TestLabel.csv", index=False, header=False)
df_target.to_csv(save_path + "TL/TestLabel.csv", index=False, header=False)
df_target.to_csv(save_path + "CL/TestLabel.csv", index=False, header=False)

for tn in target_num:
    cl_data = []
    cl_lable = []
    for lable in tl_y:
        if tn == lable:
            try:
                cl_data.append(tl_x[lable])
                cl_lable.append(lable)
            except:
                print(lable)

    df_data = pd.DataFrame(cl_data)
    df_data.to_csv(save_path + "CL/cl_TainData_" + str(tn) + ".csv", index=False, header=False)
    df_data = pd.DataFrame(cl_lable)
    df_data.to_csv(save_path + "CL/cl_TainLabel_" + str(tn) + ".csv", index=False, header=False)

print("Finish")