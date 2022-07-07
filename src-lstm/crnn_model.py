import torch
import torch.nn as nn
import torch.nn.functional as F


class PromoterNet(nn.Module):

    def __init__(self,lstm_dropout:float=0.0, cnn_dropout:float=0.2):
        super().__init__()
        # first layer num_filters = 64
        # larger kernel size
        # no lstm at the top of transformer
        # we should make sure all receptive fields are covered
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=16, stride=3)  # Feature Maps: (33, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (16, 64)
        self.drop1 = nn.Dropout(cnn_dropout)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1)  # Feature Maps: (11, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (5, 128)
        self.drop2 = nn.Dropout(cnn_dropout)
        self.lstm = nn.LSTM(input_size=128 * 2, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True, dropout=lstm_dropout)

        self.flatten = nn.Flatten()  # Feature Vectors: (5 * Hout * D, 1) = (5 * 128 * 2, 1) = (1280, 1)
        self.fc = nn.Linear(1280, 1)  # input space = lstm hidden size * 2

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))

        batch_size = x.shape[0]  # 1024
        half = int(batch_size / 2)
        feature_maps, feature_maps_comp = x[:half], x[half:]
        feature_maps_all = torch.cat((feature_maps, feature_maps_comp), dim=1)  # 512

        # feature_maps_all.shape = Batch x channel x signal
        # lstm input format = Batch x signal x channel
        feature_maps_all = torch.permute(feature_maps_all, dims=(0, 2, 1))

        # LSTM Input: N x L x Hin = 512 x 5 x (128*2)
        # LSTM Output: N x L x (D*Hout) = 512 x 5 x (2*128) [D = 2 because bi-directional]
        output, (hn, cn) = self.lstm(feature_maps_all)
        x = self.flatten(output)
        out = self.fc(x)
        return torch.squeeze(out)