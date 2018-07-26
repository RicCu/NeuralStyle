import torch
import torchvision

import visdom
from PIL import ImageColor

from collections import Iterable


class Scalar():
    def __init__(self, name, title=None, xlabel=None, ylabel=None,
                 multi_trace=False, env=None):
        self.vis = visdom.Visdom(env=env)
        if title is None:
            title = name
        opts = dict(title=title, xlabel=xlabel, ylabel=ylabel,
                    showlegend=multi_trace)
        self.win = name
        self.opts = opts
        self.multi_trace = multi_trace

    def add(self, step, data, trace=None):
        if not isinstance(step, torch.Tensor):
            if not isinstance(step, Iterable):
                step = [step]
            step = torch.Tensor([*step])
        if not self.vis.win_exists(self.win):
            self.vis.line(X=step, Y=data, name=trace, win=self.win,
                          opts=self.opts)
        if self.multi_trace and trace is None:
            raise ValueError('Set trace when using multi-trace graph')
        self.vis.line(X=step, Y=data, name=trace, win=self.win,
                      update='append', opts=self.opts)

class Image():
    def __init__(self, title, env=None):
        self.vis = visdom.Visdom(env=env)
        self.title = title + '_{}'

    def add(self, step, image):
        if not isinstance(step, torch.Tensor):
            if not isinstance(step, Iterable):
                step = [step]
            step = torch.Tensor([*step])
        if image.shape[0] == 1:
            self.vis.image(image.squeeze(0).detach(),
                           opts={'title': self.title.format(step.item())})
        else:
            self.vis.images(image.detach(),
                            opts={'title': self.title.format(step.item())})

