""" Perceptual loss functions for style transfer """

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.models import vgg16


def gram(feature_map):
    """Calculates the gram matrix of a given feature map
    Args:
        feature_maps (Variable): Feature activations of shape N*c*h*w
    Returns:
        gram matrix of shape N*channels*channels
    """
    b, c, h, w = feature_map.data.shape
    fm = feature_map.view([b, c, h * w])
    fm_t = fm.transpose(1, 2)
    gram = torch.matmul(fm, fm_t) / (c * h * w)
    return gram


class StyleLoss(nn.Module):
    def __init__(self, target_feature_map):
        super(StyleLoss, self).__init__()
        self.register_buffer('target', gram(target_feature_map.detach()))
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, x):
        b, _, _, _ = x.shape
        target = self.target.repeat(b, 1, 1)
        G = gram(x)
        self.loss = self.criterion(G, target)
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class ContentLoss(nn.Module):
    """Module to compute the content loss. Allows arbitrary size style images
    during initialization and updating the content target.
    Usage: During loss network definition set compute_loss to False, to allow,
    after initialization iterate through ContentLoss modules and set
    compute_loss to True to perform the loss evaluation at every forward pass.
    When doing optimization for multiple content targets, perform a forward
    pass with the target images and then use update() to set the target to
    those images.
    """
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, x, target):
        _, c, h, w = x.data.shape # TODO Dynamic or static into weight?
        self.loss = self.criterion(x, target.detach()) / (c*h*w)
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class TVLoss(nn.Module):
    """Implements Anisotropic Total Variation regularization"""
    def __init__(self):
        super(TVLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x):
        X = x.detach()
        XX = x#.clone()
        b, c, h, w = X.data.shape
        y_tv = self.criterion(XX[:, :, 1:, :], X[:, :, :h-1, :])
        x_tv = self.criterion(XX[:, :, :, 1:], X[:, :, :, :w-1])
        self.loss = (y_tv + x_tv) # / (c*h*w) # TODO Normalize?
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class PerceptualLoss(nn.Module):
    def __init__(self, style_activations, style_layers, content_layers,
                 style_weight, content_weight, tv_weight):
        super(PerceptualLoss, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.style_losses = dict() # TODO DIct -> Not properly registered!
        for layer in style_layers:
            self.style_losses[layer] = StyleLoss(style_activations[layer])
        self.content_losses = dict()
        for layer in content_layers:
            self.content_losses[layer] = ContentLoss()
        self.tv_loss = TVLoss()

    def forward(self, x, img, content_activations):
        style_loss = content_loss = 0.0
        for layer in self.style_layers:
            style_loss += self.style_losses[layer](x[layer])
        for layer in self.content_layers:
            content_loss += self.content_losses[layer](x[layer],
                                                content_activations[layer])
        tv_loss = self.tv_loss(img)
        style_loss *= self.style_weight
        content_loss *= self.content_weight
        tv_loss *= self.tv_weight
        loss = style_loss + content_loss + tv_loss
        return {'total_loss': loss, 'style_loss': style_loss,
                'content_loss': content_loss, 'tv_loss': tv_loss}


class VGG16(nn.Module):
    def __init__(self, relu_layers):
        super(VGG16, self).__init__()
        vgg = vgg16(pretrained=True).features
        num_blocks = len(relu_layers)
        self.relu_layers = relu_layers
        self.blocks = nn.ModuleList([nn.Sequential() for _ in
                                     range(num_blocks)])
        layer_num = 1
        block_num = 0
        for layer in list(vgg):
            if isinstance(layer, nn.Conv2d):
                name = 'conv_'+str(layer_num)
                self.blocks[block_num].add_module(name, layer)
            if isinstance(layer, nn.ReLU):
                name = 'relu_'+str(layer_num)
                self.blocks[block_num].add_module(name, layer)
                if name in relu_layers:
                    block_num += 1
                    if block_num == num_blocks - 1:
                        break
                layer_num += 1
            if isinstance(layer, nn.MaxPool2d):
                name = 'pool_'+str(layer_num + 1)
                self.blocks[block_num].add_module(name, layer)
            del layer

    def forward(self, x):
        out = dict()
        h = x
        for name, block in zip(self.relu_layers, self.blocks):
            h = block(h)
            out[name] = h
        return out

    def train(self, mode=True):
        for param in self.parameters():
            param.requires_grad = mode
        return super(VGG16, self).train(mode)

