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
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=512, kernel_size=22, stride=6)  # Feature Maps: (16, 512)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Feature Maps: (4, 512)
        self.drop1 = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size=512 * 2, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        self.flatten = nn.Flatten()  # Feature Vectors: (4 * Hout * D, 1) = (4 * 256 * 2, 1) = (2048, 1)
        self.fc = nn.Linear(2048, 1)  # input space = lstm hidden size * 2

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))

        batch_size = x.shape[0]  # 1024
        half = int(batch_size / 2)
        feature_maps, feature_maps_comp = x[:half], x[half:]
        feature_maps_all = torch.cat((feature_maps, feature_maps_comp), dim=1)  # 512

        # feature_maps_all.shape = Batch x channel x signal
        # lstm input format = Batch x signal x channel
        feature_maps_all = torch.permute(feature_maps_all, dims=(0, 2, 1))

        # LSTM Input: N x L x Hin = 512 x 10 x (128*2)
        # LSTM Output: N x L x (D*Hout) = 512 x 10 x (2*128) [D = 2 because bi-directional]
        output, (hn, cn) = self.lstm(feature_maps_all)
        x = self.flatten(output)
        out = self.fc(x)
        return torch.squeeze(out)