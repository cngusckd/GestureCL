import os
import numpy as np

# data_path = 'Data/Digit/3seconds/100Hz/'
data_path = 'Data/Target/HS/'

train_X = np.load(data_path + 'train_x.npy')
train_Y = np.load(data_path + 'train_y.npy')
test_X = np.load(data_path + 'val_x.npy')
test_Y = np.load(data_path + 'val_y.npy')

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

