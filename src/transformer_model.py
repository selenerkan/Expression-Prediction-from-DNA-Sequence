
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
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=12, stride=1)  # Feature Maps: (37, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (18, 128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8)  # Feature Maps: (11, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (5, 256)

        # d_model = size of input embeddings, nhead = the number of parallel heads, dk = dv = d_model / nhead
        # One sub_attention_fc module takes an input of NxLxD_model shape and produces output of NxLxD_model shape
        # Input(sub_attention_fc) = Nx5x256, Output(sub_attention_fc) = Nx5x256
        # Input(transformer_encoder) = Nx5x256, Output(transformer_encoder) = Nx5x256
        sub_attention_fc = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(sub_attention_fc, num_layers=3)

        # It takes the input in the shape of NxCxLin and reduces it into NxCxLout
        # C: Channel (256 is channel in our case)
        # Input(global_max_pooling) = Nx256x5, Output(global_max_pooling) = Nx256x1
        self.global_max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        batch_size = x.shape[0]  # 1024
        half = int(batch_size / 2)
        feature_maps, feature_maps_comp = x[:half], x[half:]
        feature_maps_all = torch.cat((feature_maps, feature_maps_comp), dim=1)  # 512

        # feature_maps_all.shape = Batch x channel x signal
        # transformer input format = Batch x signal x channel
        # WARNING: feature_maps_all should pass through positional encoding first
        input_signal = torch.permute(feature_maps_all, dims=(0, 2, 1))
        encoded_input_signal = self.transformer_encoder(input_signal)

        # transformer output format = Batch x signal x channel
        # global max pooling input format = Batch x channel x signal
        encoded_input_signal = torch.permute(encoded_input_signal, dims=(0, 2, 1))
        pooled_signal = self.global_max_pooling(encoded_input_signal)
        features = torch.squeeze(pooled_signal)
        out = self.fc(features)

        return torch.squeeze(out)