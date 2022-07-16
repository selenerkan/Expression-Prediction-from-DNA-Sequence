
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromoterNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=16, stride=3)  # Feature Maps: (33, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (16, 64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1)  # Feature Maps: (11, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (5, 128)

        # d_model = size of input embeddings, nhead = the number of parallel heads, dk = dv = d_model / nhead
        # One sub_attention_fc module takes an input of NxLxD_model shape and produces output of NxLxD_model shape
        # Input(sub_attention_fc) = Nx5x256, Output(sub_attention_fc) = Nx5x256
        # Input(transformer_encoder) = Nx5x256, Output(transformer_encoder) = Nx5x256
        sub_attention_fc = nn.TransformerEncoderLayer(d_model=128 * 2, nhead=4, dim_feedforward=256, batch_first=True, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(sub_attention_fc, num_layers=3)

        self.flatten = nn.Flatten()  # Feature Vectors: (5 * 256, 1) = (1280, 1)
        self.fc = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

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
        x = self.flatten(encoded_input_signal)
        out = self.fc(x)
        return torch.squeeze(out)