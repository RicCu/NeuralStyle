"""Set of utility functions to promote code reuse"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def build_common_arguments(parser):
    parser.add_argument('--cuda', action='store_true', default=False,
                        dest='device', help='Use cuda if available')
    parser.add_argument('--style-data', type=str,
                        default='.assets/styles/scream.jpg',
                        help='Style image')
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
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--log-every', type=int, default=200,
                        help='Number of steps before logging statistics')
    parser.add_argument('--log-images-every', type=int, default=200,
                        help='Number of steps before logging the generated '
                        'images')
    return parser


def set_device(params):
    if params.device:
        if torch.cuda.is_available():
            params.device = torch.device('cuda')
        else:
            print('Cuda was selected but is not available, continuing on cpu '
                  'only')
            params.device = torch.device('cpu')
    else:
        params.device = torch.device('cpu')
    return params


def load_image(path_to_image, shape=None):
    """Loads an image and preprocess it. If shape is set, preprocessing resizes
    it, otherwize it only load the image and normalizes it.
    Args:
        path_to_image (str): path to image from current directory.
        shape (list of int): Height and width to resize.
    """
    if shape is not None:
        transform = transforms.Compose([transforms.Resize(shape),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=RGB_MEAN,
                                                             std=RGB_STD)])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=RGB_MEAN,
                                                             std=RGB_STD)])
    img = transform(Image.open(path_to_image)).unsqueeze(0)
    img.requires_grad = False
    return img


def build_dataloader(path_to_data, img_shape, batch_size,
                     device=torch.device('cpu'), num_workers=2):
    """Initialize an ImageFolder DataLoader with preprocessing pipeline"""
    if device.type == 'cuda':
        pin_memory = True
    else:
        pin_memory = False
    transform = transforms.Compose([transforms.Resize(img_shape),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=RGB_MEAN,
                                                         std=RGB_STD)])
    dataset = ImageFolder(path_to_data, transform=transform)
    data = DataLoader(dataset, batch_size, True, num_workers=num_workers,
                      pin_memory=pin_memory, drop_last=True)
    return data
