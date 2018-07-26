""" Implementation of a fast-style transfer network """

import torch
import torch.nn.functional as F
import torch.nn as nn


class Residual(nn.Module):
    """Unlinke other blocks, this module receives unpadded inputs."""
    def __init__(self, channels, kernel_size=3):
        super(Residual, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.pad = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size)
        self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        h = self.pad(x)
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.pad(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = h + x
        return h


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1,
                 use_relu=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        self.batch_norm = nn.InstanceNorm2d(out_channels)
        self.use_relu = use_relu

    def forward(self, x):
        h = self.conv(x)
        h = self.batch_norm(h)
        if self.use_relu:
            h = F.relu(h)
        return h


class ConvTran(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTran, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1,
                                         1)
        self.batch_norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        h = self.conv_t(x)
        h = self.batch_norm(h)
        return F.relu(h)


class FastStyle(nn.Module):
    def __init__(self):
        super(FastStyle, self).__init__()
        self.conv1 = Conv(3, 32, 9)
        self.conv2 = Conv(32, 64, 3, 2)
        self.conv3 = Conv(64, 128, 3, 2)
        self.res1 = Residual(128)
        self.res2 = Residual(128)
        self.res3 = Residual(128)
        self.res4 = Residual(128)
        self.res5 = Residual(128)
        self.convT1 = ConvTran(128, 64)
        self.convT2 = ConvTran(64, 32)
        self.conv_out = Conv(32, 3, 9, use_relu=False)
        self._init()

    def forward(self, x):
        def reflect_padding(x, f, s, half=False):
            if half:
                denom = 2
            else:
                denom= 1
            _, _, h, w = x.data.shape
            pad_w = (w * ((s/denom) - 1) + f - s)
            pad_h = (h * ((s/denom) - 1) + f - s)
            if pad_w % 2 == 1:
                pad_l = int(pad_w//2) + 1
                pad_r = int(pad_w//2)
            else:
                pad_l = pad_r = int(pad_w / 2)
            if pad_h % 2 == 1:
                pad_t = int(pad_h//2) + 1
                pad_b = int(pad_h//2)
            else:
                pad_t = pad_b = int(pad_h / 2)
            return F.pad(x, [pad_l, pad_r, pad_t, pad_b], mode='reflect')
        h = self.conv1(reflect_padding(x, 9, 1))
        h = self.conv2(reflect_padding(h, 3, 2, True))
        h = self.conv3(reflect_padding(h, 3, 2, True))
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.convT1(h)
        h = self.convT2(h)
        h = self.conv_out(reflect_padding(h, 9, 1))
        h = F.tanh(h) * 0.5 + 0.5
        return h

    def _init(self):
        def __init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                #nn.init.normal(m.weight.data, std=0.1)
                nn.init.constant_(m.bias.data, 0) # TODO replace with zero!
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                #nn.init.normal(m.weight.data, std=0.1)
                nn.init.constant_(m.bias.data, 0)
        self.apply(__init)


