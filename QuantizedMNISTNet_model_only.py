"""
Minimal SmallMNISTNet definition without top-level training/download side effects.
Use with convert_model_to_rtl.py to load pretrained weights quickly.
"""

import torch.nn as nn
import torch.nn.functional as F


class SmallMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
