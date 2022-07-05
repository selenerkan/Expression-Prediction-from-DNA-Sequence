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
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=16, stride=1)  # Feature Maps: (97, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (48, 64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1)  # Feature Maps: (43, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (21, 128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)  # Feature Maps: (19, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (9, 256)
        # encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
        #                                             activation, layer_norm_eps, batch_first, norm_first,
        #                                             **factory_kwargs)
        # encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.flatten = nn.Flatten()  # Feature Vectors: (2304, 1)
        self.fc1 = nn.Linear(2304 * 2, 1152)
        self.fc2 = nn.Linear(1152, 1)

        """self.conv4 = nn.Conv2d(in_channels=2304 * 2, out_channels=2304, kernel_size=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=2304, out_channels=1152, kernel_size=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=1152, out_channels=1, kernel_size=(1, 1))"""

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
        out = self.fc2(x)

        return torch.squeeze(out)