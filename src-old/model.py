import torch.nn as nn
import torch.nn.functional as F


class PromoterNet(nn.Module):

    def __init__(self, conv_layers=[5,8,16], fc_layers=[80,32,16], kernels=[8,4,4,4], strides=[1,4,1,4]):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=conv_layers[0], out_channels=conv_layers[1], kernel_size=kernels[0], stride=strides[0])  # Feature Maps: (105, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=kernels[1], stride=strides[1])  # Feature Maps: (26, 16)
        self.conv2 = nn.Conv1d(in_channels=conv_layers[1], out_channels=conv_layers[2], kernel_size=kernels[2], stride=strides[2])  # Feature Maps: (23, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=kernels[3], stride=strides[3])  # Feature Maps: (5, 64)
        self.flatten = nn.Flatten()  # Feature Vectors: (320, 1)
        self.adapt_pool = nn.AdaptiveMaxPool1d(fc_layers[0]) # Ensures the output size is always the fixed size.
        self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc2 = nn.Linear(fc_layers[1], fc_layers[2])
        self.fc3 = nn.Linear(fc_layers[2], 1)

    def forward(self, seq):
        x = self.pool1(F.relu(self.conv1(seq)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)

        x = self.adapt_pool(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.relu(self.fc3(x))

        return out.squeeze(1) # remove feature dimension since there is only one target label.
