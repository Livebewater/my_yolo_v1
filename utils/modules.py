import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, leakyrelu=False):
        super(Conv2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyrelu else nn.ReLU(inplace=True)
        )

    def forward(self, data):
        return self.layer(data)


class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.layer(data) * data


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPP, self).__init__()
        self.fuse_conv = Conv2d(in_channels * 4, out_channels, 1, leakyrelu=True)

    def forward(self, data):
        x_1 = F.max_pool2d(data, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(data, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(data, 3, stride=1, padding=6)
        x = torch.cat([data, x_1, x_2, x_3], dim = 1)
        return self.fuse_conv(x)
