import numpy as np
import os
# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

signals_path = 'har_uci/train/Inertial Signals'
signals_paths = [os.path.join(signals_path, i) for i in os.listdir(signals_path)]

x_train = load_X(signals_paths)
y_train = load_y('har_uci/train/y_train.txt')

print('x train shape : ', x_train.shape)
print('y train shape : ', y_train.shape)

signals_path = signals_path.replace('train', 'test')
signals_paths = [os.path.join(signals_path, i) for i in os.listdir(signals_path)]

x_test = load_X(signals_paths)
y_test = load_y('har_uci/test/y_test.txt')

print('x test shape : ', x_test.shape)
print('y test shape : ', y_test.shape)