"""Export pretrained models into ONNX format"""

import argparse
import os

import torch

import utils
from model import FastStyle
from model import TextureNetwork


INPUT_NAME = ['original',]
OUTPUT_NAME = ['transformed',]

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None,
                        help='Dummy image to use as input during export. '
                        'Leave empty to use a random tensor.')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint. Must include the '
                        'directory that contains the file, it will be used '
                        'as the export\'s name')
    parser.add_argument('--shape', nargs='+', type=int,
                        help='Height x width')
    parser.add_argument('--texture', default=False, action='store_true',
                        help='Set flag to use a Texture Network')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                        help='Scale of the noise tensor in Texture Networks. '
                        'This only has an effect if --texture is set')
    params = parser.parse_args()

    assert (params.data is not None or params.shape is not None), ('Set at '
                                                               'least shape '
                                                               'or data')
    if params.shape is not None:
        assert len(params.shape) == 2, 'Shape must include height and width'
    assert params.ckpt is not None, 'Path to checkpoint file must be provided'
    assert '/' in params.ckpt, ('Checkpoint path must include at least the '
                                'parent directory of the file')
    if params.data is None:
        data = torch.randn(1, 3, *params.shape)
    else:
        data = utils.load_image(params.data, params.shape)
    if params.texture:
        raise NotImplementedError('upsample_nearest2d op cannot be exported '
                                  'yet')
        model = TextureNetwork(noise_scale=params.noise_scale)
    else:
        model = FastStyle()

    # Get model name from checkpoint directory
    model_name = params.ckpt.split('/')[-2]

    utils.maybe_create_dir('onnx')
    onnx_path = os.path.join('onnx', model_name + '.onnx')
    model.load_state_dict(torch.load(params.ckpt))
    torch.onnx.export(model, data, onnx_path, input_names=INPUT_NAME,
                      output_names=OUTPUT_NAME, verbose=False)


if __name__=='__main__':
    main()
