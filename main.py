"""Main script from which to launch style transfer jobs"""

import argparse

import torch

import descriptive
import generative
import utils

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False,
                        dest='device', help='Use cuda if available')
    parser.add_argument('--descriptive', action='store_true', default=False,
                        help='Perform descriptive style transfer')
    parser.add_argument('--generative', action='store_true', default=False,
                        help='Perform generative style transfer')
    parser.add_argument('--style-data', type=str, default='styles/scream.jpg',
                        help='Style image')
    parser.add_argument('--content-data', type=str, default='chicago.jpg',
                        help='Content image (directory or image)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--shape', default=[256, 256], nargs='+', type=int,
                        help='Height x width')
    parser.add_argument('--content-weight', type=float, default=1e4,
                        help='Weight of content loss')
    parser.add_argument('--style-weight', type=float, default=1e4,
                        help='Weight of style loss')
    parser.add_argument('--tv-weight', type=float, default=1.0,
                        help='Weight of total variation denoising loss')
    parser.add_argument('--name', type=str, default='scream',
                        help='Name of the style, used to save transformed '
                        'images and checkpoints')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for generative methods')
    parser.add_argument('-lr', '--learning-rate', type=int, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--log-every', type=int, default=200,
                        help='Number of steps before logging statistics')
    parser.add_argument('--log-images-every', type=int, default=500,
                        help='Number of steps before logging the generated '
                        'images')
    parser.add_argument('--texture', default=False, action='store_true',
                        help='Set flag to use a Texture Network')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                        help='Scale of the noise tensor in Texture Networks. '
                        'This only has an effect if --texture is set')
    return parser


def main():
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS = utils.set_device(FLAGS)
    if FLAGS.descriptive:
        descriptive.train(FLAGS)
    if FLAGS.generative:
        generative.train(FLAGS)


if __name__=='__main__':
    main()

