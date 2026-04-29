import torch
from torch.nn import Module, AdaptiveAvgPool1d
from model.CNNFeatureExtractor import CNNFeatureExtractor
from model.ResidualMamba import ResidualMamba
from model.MLPConnector import MLPConnector

class KeyExtractor(Module):
    def __init__(self, batch_size=1, key_size=128, pad_value=0):
        super(KeyExtractor, self).__init__()

        self.pad_value = pad_value

        self.conv1 = CNNFeatureExtractor(in_channels=1, filters=64, kernel_size=11, stride=2, padding=5)
        self.conv2 = CNNFeatureExtractor(in_channels=64, filters=128, kernel_size=11, stride=1, padding=5)
        self.conv3 = CNNFeatureExtractor(in_channels=128, filters=256, kernel_size=11, stride=1, padding=5)
        self.conv4 = CNNFeatureExtractor(in_channels=256, filters=512, kernel_size=11, stride=1, padding=5)

        self.mamba = ResidualMamba()
        self.mlp_classifier = MLPConnector(batch_size=batch_size)


    def forward(self, x):

        mask = (x != self.pad_value).to(torch.float32)

        # Conv Module
        x, mask = self.conv1(x, mask)
        x, mask = self.conv2(x, mask)
        x, mask = self.conv3(x, mask)
        x, mask = self.conv4(x, mask)

        # Residual Mamba Module
        x = x.transpose(-2, -1)
        mask = mask.transpose(-2, -1)

        block1, mask = self.mamba(x, mask)
        block2, mask = self.mamba(block1, mask)

        block2 = block2 * mask
        pooled_block = block2.mean(dim=1)

        # MLP Connector

        y_hat = self.mlp_classifier(pooled_block)

        return y_hat













