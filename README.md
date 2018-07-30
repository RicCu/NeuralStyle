# NeuralStyle
This repository includes implementations for descriptive and generative style transfer techniques. The algorithms are generally
based on the originals introduced in [2] and [3], however it's not intended to reproduce the exact results of either paper, 
thus some hyperparameters and other implementation details have been changed.

## Dependencies
This project uses [PyTorch 0.4.1](https://github.com/pytorch/pytorch/releases/tag/v0.4.1) and 
[Visdom (8c6322d)](https://github.com/facebookresearch/visdom) for training visualizations. Please refer to their corresponding
project websites for installation instructions.

## Usage
The scripts `descriptive.py` and `generative.py` may be used to iteratively transform a single image or train a neural network,
respectively. Common arguments include:
- `--cuda` use single gpu if available
- `--style-data` path to target style image
- `--content-data` path to a directory of directories with images (*generative*) or a single image (*descriptive*)
- `--shape` height and width for content data (optional for descriptive algorithm)
- `--content-weight` weight for content-component loss
- `--style-weight` weight for style-component loss
- `--tv-weight` weight for total variation-component loss
- `--name` name for checkpoints or transformed images
- `-lr` learning rate
- `--epochs` number of iterations, epochs over all training examples (*generative*) or optimization steps (*descriptive*)
- `--log-every` step frequency to log statistics
- `--log-images-every` step frequency to log transformed images

**descriptive.py** also has the option:
- `--save-directory` path to directory where transformed images will be saved

**generative.py** also has the options:
- `--eval` use to transform an image with a pretrained model
- `--model` path to pretrained model (used with `--eval`)

For live visualizations you must have a local visdom server running: `python -m visdom.server`.

**Examples**
```
python generative.py --cuda --content-data /home/username/datasets/coco/images --style-data .assets/styles/wave.jpg --name wave --batch-size 16 --content-weight 1 --style-weight 5 --tv-weight 7 --epochs 2 

python descriptive.py --content-data .assets/guadalajara.jpg --style-data .assets/styles/wave.jpg --name wave --log-every 2 --log-images-every 3 
```

Alternatively, the script `main.py` may be used with the flags `--descriptive` or `--generative`.

## References
[1] Jing, et al. (2018). *Neural Style Transfer: A Review*. arXiv:1705.04058v6

[2] Gatys L., Ecker A., Bethge M. (2015). *A Neural Algorithm of Artistic Style*. arXiv:1508.06576v2

[3] Johnson J., Alahi A., Fei-Fei L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. arXiv:1603.08155v1