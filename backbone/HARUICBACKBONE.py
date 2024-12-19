from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.functional import avg_pool2d, relu

from backbone import MammothBackbone, register_backbone

class RT_CNN(MammothBackbone):
    def __init__(self, class_num):
        super(RT_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(9, 512, kernel_size = 7, padding = 'same')
        self.maxpool1 = nn.MaxPool1d(3, 3)

        
        self.conv2 = nn.Conv1d(512, 512, kernel_size = 7, padding = 'same')
        self.maxpool2 = nn.MaxPool1d(3, 3)

        self.conv3 = nn.Conv1d(512, 512, kernel_size = 7, padding = 'same')
        self.maxpool3 = nn.MaxPool1d(3, 3)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.Dropout(0.05),
            nn.Linear(512, class_num)
        )
    
    def forward(self, x, returnt='out'):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.fc(x)

        return x

@register_backbone("harucibackbone")
def harucibackbone():
    return RT_CNN(6)