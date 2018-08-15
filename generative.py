"""Generative neural style transfer

These implementations generally follow the original papers, but do not
intend to exactly replicate the architectures, hyperparameters or
results.

    Johnson J., Alahi A., Fei-Fei L. (2016). Perceptual Losses for
        Real-Time Style Transfer and Super-Resolution.
        arXiv:1603.08155v1
    Ulyanov D., et al. (2016). Texture Networks: Feed-forward
        Synthesis of Textures and Stylized Images.
        arXiv:1603.03417v1 [cs.CV]
"""


from time import sleep, time

import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

import monitor
import utils
from loss import *
from model import FastStyle
from model import TextureNetwork


CONTENT_LAYERS = ['relu_4']
STYLE_LAYERS = ['relu_2', 'relu_4', 'relu_7', 'relu_10']

LOSSES_WIN = 'LOSSES'
TOTAL_LOSS_WIN = 'LOSS'
CONTENT_LOSS_TRACE = 'content'
STYLE_LOSS_TRACE = 'style'
TV_LOSS_TRACE = 'tv'


def train(params):
    """Prepares inputs, networks and visualizations and trains a generative
    style transfer model
    Args:
        params (object): Holds all the necessary parameters top define the
            trainig regime
    """
    MODEL_DIRECTORY = 'ckpt/{}'.format(params.name)
    utils.maybe_create_dir(MODEL_DIRECTORY)

    # Prepare visualizations
    total_loss_graph = monitor.Scalar(TOTAL_LOSS_WIN, title='Total loss',
                                      xlabel='step', ylabel='loss',
                                      env=params.name)
    losses_graph = monitor.Scalar(LOSSES_WIN, title='Losses', xlabel='step',
                                  ylabel='loss', multi_trace=True,
                                  env=params.name)
    image_logger = monitor.Image('Transformed', env=params.name)

    # Prepare inputs and networks
    data = utils.build_dataloader(params.content_data, params.shape,
                                  params.batch_size, params.device)
    style_img = utils.load_image(params.style_data).to(params.device)

    if params.texture:
        net = TextureNetwork(noise_scale=params.noise_scale).to(params.device)
    else:
        net = FastStyle().to(params.device)

    vgg = VGG16(STYLE_LAYERS).to(params.device)
    vgg.eval()

    # Prepare loss functions and optimizer
    style_activations = vgg(style_img)
    lossFn = PerceptualLoss(style_activations, STYLE_LAYERS, CONTENT_LAYERS,
                            params.style_weight, params.content_weight,
                            params.tv_weight).to(params.device)
    del style_img
    del style_activations
    torch.cuda.empty_cache()
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # Training loop
    step = 0
    for epoch in range(1, params.epochs + 1):
        start = time()
        total_score = style_score = content_score = tv_score = 0
        for content, _ in data:
            content = content.to(params.device)
            optimizer.zero_grad()
            image = net(content)
            activations = vgg(image)
            content.requires_grad = False
            with torch.no_grad():
                content_activations = vgg(content)
            losses = lossFn(activations, image, content_activations)
            losses['total_loss'].backward()
            optimizer.step()

            # Track avg. losses
            total_score += losses['total_loss']
            style_score += losses['style_loss']
            content_score += losses['content_loss']
            tv_score += losses['tv_loss']

            # Log avg. losses
            if step % params.log_every == 0:
                total_score = total_score.unsqueeze(0) / params.log_every
                style_score = style_score.unsqueeze(0) / params.log_every
                content_score = content_score.unsqueeze(0) / params.log_every
                tv_score = tv_score.unsqueeze(0) / params.log_every
                total_loss_graph.add(step, total_score.detach())
                losses_graph.add(step, style_score.detach(), STYLE_LOSS_TRACE)
                losses_graph.add(step, content_score.detach(),
                                 CONTENT_LOSS_TRACE)
                losses_graph.add(step, tv_score.detach(), TV_LOSS_TRACE)
                print('Epoch {}, global step {}, total loss {:.5f}, '
                      'style loss {:.5f}, content loss {:.5f}, '
                      'tv loss {:.5f}, time {}'.format(epoch, step,
                                                       total_score.item(),
                                                       style_score.item(),
                                                       content_score.item(),
                                                       tv_score.item(),
                                                       time() - start))
                total_score = 0
                style_score = 0
                content_score = 0
                tv_score = 0
                start = time()

            # Log transformed images
            if step % params.log_images_every == 0:
                image_logger.add(step, image)
            step += 1

        # Save model after each epoch
        torch.save(net.state_dict(), '{}/model_{}.pt'.format(MODEL_DIRECTORY,
                                                             epoch))


def eval(params):
    """Use a pretrained model to appply style to a given image."""
    with torch.no_grad():
        content_img = utils.load_image(params.content_data)

        if params.texture:
            net = TextureNetwork(noise_scale=params.noise_scale)
        else:
            net = FastStyle()

        net.load_state_dict(torch.load(params.model))
        img = net(content_img).squeeze(0)
        img_name = params.name + '_' + params.content_data.split('/')[-1]
        torchvision.utils.save_image(img, img_name)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser = utils.build_common_arguments(parser)
    parser.add_argument('--content-data', type=str,
                        help='Directory of content images')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs to train')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Use a saved model to transform a content image')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model for fast style transfer')
    parser.add_argument('--texture', default=False, action='store_true',
                        help='Set flag to use a Texture Network')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                        help='Scale of the noise tensor in Texture Networks. '
                        'This only has an effect if --texture is set')
    params = parser.parse_args()
    if params.eval:
        if params.model is None:
            parser.error('A pretrained model must be provided during '
                         'evaluation')
        eval(params)
    else:
        params = utils.set_device(params)
        train(params)


if __name__=='__main__':
    main()
