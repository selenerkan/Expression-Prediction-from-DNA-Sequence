import torch
import torch.nn as nn
import torch.nn.functional as F


class PromoterNet(nn.Module):

    def __init__(self):
        super().__init__()
        # first layer num_filters = 64
        # larger kernel size
        # no lstm at the top of transformer
        # we should make sure all receptive fields are covered
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=10, stride=1)  # Feature Maps: (103, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (51, 16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=6, stride=1)  # Feature Maps: (46, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (23, 64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)  # Feature Maps: (21, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (10, 128)
        self.flatten = nn.Flatten()  # Feature Vectors: (1280, 1)
        self.fc1 = nn.Linear(1280 * 2, 1280)
        self.fc2 = nn.Linear(1280, 640)
        self.fc3 = nn.Linear(640, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)

        batch_size = x.shape[0]  # 1024
        half = int(batch_size / 2)
        features, features_comp = x[:half], x[half:]
        features_all = torch.cat((features, features_comp), dim=1)  # 512

        x = F.relu(self.fc1(features_all))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return torch.squeeze(out)