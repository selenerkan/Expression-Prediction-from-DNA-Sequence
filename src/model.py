import torch.nn as nn
import torch.nn.functional as F


class PromoterNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=8, stride=1)  # Feature Maps: (105, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Feature Maps: (26, 16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=4, stride=1)  # Feature Maps: (23, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # Feature Maps: (5, 64)
        self.flatten = nn.Flatten()  # Feature Vectors: (320, 1)
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 1)

    def forward(self, seq):
        x = self.pool1(F.relu(self.conv1(seq)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.relu(self.fc3(x))

        return out
