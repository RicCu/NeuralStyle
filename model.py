"""Architecture for Real-Time style transfer.

These implementations generally follow the original papers, but do not
intend to exactly replicate the architectures, hyperparameters or
results.

    Gatys L., Ecker A., Bethge M. (2015). A Neural Algorithm of
        Artistic Style. arXiv:1508.06576v2
    Johnson J., Alahi A., Fei-Fei L. (2016). Perceptual Losses for
        Real-Time Style Transfer and Super-Resolution.
        arXiv:1603.08155v1
"""


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
    """Convolutional block. 2d-conv -> batch norm -> (optionally) relu"""

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
        h = torch.tanh(h) * 0.5 + 0.5
        return h

    def _init(self):
        def __init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0) # TODO replace with zero!
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        self.apply(__init)


class _TextureConvGroup(nn.Module):
    """Group of 3 convolutional blocks.

    1.- reflect_padding()
    2.- Conv(in_channels, out_channels, kernel=3, use_relu=False)
    3.- LeakyReLU()
    4.- reflect_padding()
    5.- Conv(out_channels, out_channels, kernel=3)
    6.- LeakyReLU()
    7.- reflect_padding()
    8.- Conv(out_channels, out_channels, kernel=1)
    9.- LeakyReLU()
    """

    def __init__(self, in_channels, out_channels):
        super(_TextureConvGroup, self).__init__()
        self.block1 = Conv(in_channels, out_channels, 3, use_relu=False)
        self.block2 = Conv(out_channels, out_channels, 3, use_relu=False)
        self.block3 = Conv(out_channels, out_channels, 1, use_relu=False)

    def forward(self, x):
        h = reflect_padding(x, 3, 1)
        h = self.block1(h)
        h = F.leaky_relu(h)
        h = reflect_padding(h, 3, 1)
        h = self.block2(h)
        h = F.leaky_relu(h)
        h = reflect_padding(h, 1, 1)
        h = self.block3(h)
        h = F.leaky_relu(h)
        return h


class _TextureJoinBlock(nn.Module):
    """Joins activations from two distinct sizes"""

    def __init__(self, in_channels_small, in_channels_large):
        super(_TextureJoinBlock, self).__init__()
        self.bn_small = nn.BatchNorm2d(in_channels_small)
        self.bn_large = nn.BatchNorm2d(in_channels_large)

    def forward(self, x):
        """X (list) <-- [x_small, x_large]"""
        x_small, x_large = x
        x_small = self.bn_small(F.interpolate(x_small, x_large.shape[2:]))
        x_large = self.bn_large(x_large)
        return torch.cat([x_small, x_large], dim=1)


class TextureNetwork(nn.Module):

    def __init__(self, num_scales=6, base_num_channels=8, noise_scale=1):
        super(TextureNetwork, self).__init__()
        self.num_scales = num_scales - 1
        self.noise_scale = noise_scale
        self.img_blocks = nn.ModuleList()
        for _ in range(num_scales):
            self.img_blocks.append(_TextureConvGroup(4, base_num_channels))
        self.second_blocks = nn.ModuleList()
        pre_num_channels = base_num_channels
        for _ in range(num_scales - 1):
            num_channels = pre_num_channels + base_num_channels
            self.second_blocks.append(
                nn.Sequential(
                    _TextureJoinBlock(pre_num_channels, base_num_channels),
                    _TextureConvGroup(num_channels, num_channels)
                    )
                )
            pre_num_channels = num_channels
        self.out = Conv(num_channels, 3, kernel=1, use_relu=False)
        self.noise_dist = torch.distributions.Uniform(low=0.0, high=1.0)

    def forward(self, img):
        x = self._preprocess_image(img, self.num_scales)
        h_small = self.img_blocks[0](x)
        for i in range(0, self.num_scales):
            x = self._preprocess_image(img, self.num_scales - i - 1)
            h_large = self.img_blocks[i + 1](x)
            h_small = self.second_blocks[i]([h_small, h_large])
        h = reflect_padding(h_small, 1, 1)
        return torch.tanh(self.out(h)) * 0.5 + 0.5

    def _preprocess_image(self, img, image_scale):
        b, _, h, w = img.shape
        h, w = int(h / (2**image_scale)), int(w / (2**image_scale))
        z = self.noise_dist.sample(torch.Size([b, 1, h, w]))
        z = self.noise_scale * z
        return torch.cat([F.interpolate(img, [h, w]), z.to(img)], dim=1)


def reflect_padding(x, f, s, half=False):
    if half:
        denom = 2
    else:
        denom = 1
    _, _, h, w = x.shape
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
