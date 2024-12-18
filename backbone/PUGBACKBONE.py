from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu

from backbone import MammothBackbone, register_backbone

class CNN_2Stream(MammothBackbone):
    def __init__(self, class_num):
        super(CNN_2Stream, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 512, kernel_size = 7, padding = 'same')
        self.maxpool1 = nn.MaxPool1d(3, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv1d(512, 512, kernel_size = 7, padding = 'same')
        self.maxpool2 = nn.MaxPool1d(3, 3)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv1d(512, 512, kernel_size = 7, padding = 'same')
        self.maxpool3 = nn.MaxPool1d(3, 3)
        self.relu2 = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 512),
            nn.Dropout(0.05),
            nn.Linear(512, class_num)
        )
    
    def forward(self, x, returnt='out'):
        x = x.squeeze()

        x1 = x[:, :3, :]

        x1 = self.conv1(x1)
        x1 = self.maxpool1(x1)
        x1 = self.sigmoid(x1)
        x1 = self.conv2(x1)
        x1 = self.maxpool2(x1)
        x1 = self.relu1(x1)
        x1 = self.conv3(x1)
        x1 = self.maxpool3(x1)
        x1 = self.relu2(x1)

        x2 = x[:, 3:, :]
        x2 = self.conv1(x2)
        x2 = self.maxpool1(x2)
        x2 = self.sigmoid(x2)
        x2 = self.conv2(x2)
        x2 = self.maxpool2(x2)
        x2 = self.relu1(x2)
        x2 = self.conv3(x2)
        x2 = self.maxpool3(x2)
        x2 = self.relu2(x2)

        x = torch.concat([x1, x2], dim = 2)
        x = self.fc(x)

        return x
@register_backbone("pugbackbone")
def pugbackbone():
    return CNN_2Stream(4)