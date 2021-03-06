"""Descriptive neural style transfer between images.

These implementations generally follow the original papers, but do not
intend to exactly replicate the architectures, hyperparameters or
results.

    Gatys L., Ecker A., Bethge M. (2015). A Neural Algorithm of
        Artistic Style. arXiv:1508.06576v2
"""


import torch
import torchvision
from torch import optim
from PIL import Image

import monitor
import utils
from loss import VGG16, PerceptualLoss


CONTENT_LAYERS = ['relu_4']
STYLE_LAYERS = ['relu_2', 'relu_4', 'relu_7', 'relu_10']

LOSSES_WIN = 'LOSSES'
LOSS_WIN = 'LOSS'
CONTENT_LOSS_TRACE = 'content'
STYLE_LOSS_TRACE = 'style'
TV_LOSS_TRACE = 'tv'


def train(params):
    # Prepare visualizations and directory
    total_loss_graph = monitor.Scalar(LOSS_WIN, title='Total loss',
                                      xlabel='epoch', ylabel='loss',
                                      env=params.name)
    losses_graph = monitor.Scalar(LOSSES_WIN, title='Losses', xlabel='step',
                                  ylabel='loss', env=params.name)
    image_logger = monitor.Image('Transfomed', env=params.name)
    utils.maybe_create_dir(params.save_directory)

    # Prepare inputs and feature extraction network
    content_img = utils.load_image(params.content_data,
                                   params.shape).to(params.device)
    style_img = utils.load_image(params.style_data).to(params.device)
    if params.shape is None:
        params.shape = content_img.shape[2:]
    image = torch.empty(1, 3, *params.shape,
                        device=params.device).uniform_().requires_grad_()
    vgg = VGG16(STYLE_LAYERS).to(params.device)
    vgg.eval()

    # Prepare loss function and optimizer
    style_activations = vgg(style_img)
    content_activations = vgg(content_img)
    loss_fn = PerceptualLoss(style_activations, STYLE_LAYERS, CONTENT_LAYERS,
                             params.style_weight, params.content_weight,
                             params.tv_weight).to(params.device)
    # TODO LBFGS gets stuck, must investigate
    # optimizer = optim.LBFGS([image], lr=params.learning_rate)
    optimizer = optim.Adam([image], lr=params.learning_rate)
    del style_img
    del content_img
    del style_activations

    # Store statistics
    style_score = list()
    content_score = list()
    tv_score = list()

    # Optimization loop
    for epoch in range(1, params.epochs + 1):
        def closure():
            image.data.clamp_(0, 1)
            optimizer.zero_grad()
            losses = loss_fn(vgg(image), image, content_activations)
            losses['total_loss'].backward()
            # Store statistics
            style_score.append(losses['style_loss'].item())
            content_score.append(losses['content_loss'].item())
            tv_score.append(losses['tv_loss'].item())
            return losses['total_loss']

        optimizer.step(closure)

        # Log avg. losses (through epochs AND LBFGS iterations)
        if epoch % params.log_every == 0:
            style_score = torch.tensor(style_score).mean().unsqueeze(0)
            content_score = torch.tensor(content_score).mean().unsqueeze(0)
            tv_score = torch.tensor(tv_score).mean().unsqueeze(0)
            total_loss_graph.add(epoch, (style_score + content_score +
                                         tv_score).detach())
            losses_graph.add(epoch, style_score.detach(), STYLE_LOSS_TRACE)
            losses_graph.add(epoch, content_score.detach(), CONTENT_LOSS_TRACE)
            losses_graph.add(epoch, tv_score.detach(), TV_LOSS_TRACE)
            print('Epoch {}, style loss {:.5f}, '
                  'content loss {:.5f}, tv loss {:.5f}'.format(
                      epoch,
                      style_score.item(),
                      content_score.item(),
                      tv_score.item()))
            style_score = list()
            content_score = list()
            tv_score = list()

        # Log AND save transformed images
        if epoch % params.log_images_every == 0:
            image_logger.add(epoch, image)
            file_name = '{}/{}_{}.jpg'.format(params.save_directory,
                                              params.name, epoch)
            torchvision.utils.save_image(image, file_name)

    # Save the image of the last epoch
    file_name = '{}/{}_{}.jpg'.format(params.save_directory, params.name,
                                      epoch)
    torchvision.utils.save_image(image, file_name)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser = utils.build_common_arguments(parser)
    parser.add_argument('--content-data', type=str,
                        default='.assets/guadalajara.jpg',
                        help='Content image to transform')
    parser.add_argument('--save-directory', type=str, default='transformed',
                        help='Directory to store resulting images')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to optimize')
    parser.add_argument('--shape', nargs='+', type=int,
                        help='Height x width')
    params = parser.parse_args()
    params = utils.set_device(params)
    train(params)


if __name__=='__main__':
    main()

