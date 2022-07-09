import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        #self.predicted_exp = 0

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=16, stride=1)  # Feature Maps: (97, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (48, 64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=12, stride=1)  # Feature Maps: (37, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (18, 128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8)  # Feature Maps: (11, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (5, 256)

        #self.fc_exp = nn.Linear(1280 * 2, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        """batch_size = x.shape[0]  # 1024
        half = int(batch_size / 2)
        features, features_comp = x[:half], x[half:]
        features_all = torch.cat((features, features_comp), dim=1)  # 512
        self.predicted_exp = self.fc_exp(features_all)"""

        return x


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.upsample1 = nn.Upsample(size=(256, 18), mode="bicubic")  # Input: (5, 256) Output: (18, 256)
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8, stride=1, padding=1)  # Input: (18, 256) Output: (18, 128)
        self.upsample2 = nn.Upsample(size=(128, 48), mode="bicubic")  # Input: (18, 128) Output: (48, 128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=12, stride=1, padding=1)  # Input: (48, 128) Output: (48, 64)
        self.upsample3 = nn.Upsample(size=(64, 112), mode="bicubic")  # Input: (48, 64) Output: (112, 64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=5, kernel_size=8, stride=1, padding=1)  # Input: (112, 64) Output: (112, 5)

    def forward(self, x):

        x = F.relu(self.conv1(self.upsample1(x)))
        x = F.relu(self.conv2(self.upsample2(x)))
        decoded = F.relu(self.conv3(self.upsample3(x)))

        return decoded


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x