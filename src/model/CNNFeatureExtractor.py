from torch.nn import Module, Conv1d, BatchNorm1d, AvgPool1d, ReLU, LeakyReLU, MaxPool1d
import torch.nn.functional as F
from src.CauchyActivation import CauchyActivation


class CNNFeatureExtractor(Module):
    def __init__(self, in_channels, filters, kernel_size=1, stride=1, padding=0, pool_size=2):
        super(CNNFeatureExtractor, self).__init__()

        self.conv = Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = BatchNorm1d(in_channels)
        self.avg_pool = AvgPool1d(kernel_size=pool_size, stride=stride*2, count_include_pad=False)
        self.max_pool = MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, mask):

        x = self.batch_norm(x)
        x = self.conv(x)
        x = LeakyReLU()(x)

        mask = self.max_pool(mask)

        x = x * mask
        x = self.avg_pool(x)
        mask = self.avg_pool(mask)

        return x, mask