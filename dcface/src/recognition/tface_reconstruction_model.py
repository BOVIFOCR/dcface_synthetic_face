import argparse
import os, sys
import random
from glob import glob
from pathlib import Path
import time




def get_MICA(input_size):
    import cv2
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face
    from insightface.utils import face_align
    from loguru import logger
    from pytorch3d.io import save_ply, save_obj
    from skimage.io import imread
    from tqdm import tqdm

    def get_args_defaults():
        parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
        parser.add_argument('-m', default='data/pretrained/mica.tar', type=str, help='Pretrained model path')
        args = parser.parse_args()
        return args

    def load_checkpoint_MICA(weights, mica):
        checkpoint = torch.load(weights)
        if 'arcface' in checkpoint:
            mica.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            mica.flameModel.load_state_dict(checkpoint['flameModel'])

    # from configs.config import get_cfg_defaults
    from MICA.configs.config import get_cfg_defaults
    from MICA.datasets.creation.util import get_arcface_input, get_center
    from MICA.utils import util

    # print('    getting default MICA args')
    # args = get_args_defaults()
    print('getting default MICA configs')
    cfg = get_cfg_defaults()
    mica_weights = 'src/MICA/data/pretrained/mica.tar'

    device = 'cuda:0'
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='MICA.micalib.models', model_name=cfg.model.name)(cfg, device)
    print('loading MICA checkpoint:', mica_weights)
    load_checkpoint_MICA(mica_weights, mica)
    mica.eval()
    return mica





def get_BFM(input_size=(224, 224)):
    import torch
    from models.hrn import Reconstructor

    # def load_checkpoint_BFM(weights, mica):
    #     checkpoint = torch.load(weights)
    #     if 'arcface' in checkpoint:
    #         mica.arcface.load_state_dict(checkpoint['arcface'])
    #     if 'flameModel' in checkpoint:
    #         mica.flameModel.load_state_dict(checkpoint['flameModel'])

    params = [
        '--checkpoints_dir', 'src/HRN/assets/pretrained_models',
        '--name', 'hrn_v1.1',
        '--epoch', '10',
    ]

    print('loading BFM:', params[1])
    reconstructor = Reconstructor(params)
    return reconstructor







# based on:
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/backbone/model_irse.py
'''
from collections import namedtuple
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Module

import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid
from torch.nn import Module
import numpy as np

from torch.nn import PReLU
'''

'''
def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
'''

'''
class Flatten(Module):
    """ Flat tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)
'''

'''
class LinearBlock(Module):
    """ Convolution block without no-linear activation layer
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
'''

'''
class GNAP(Module):
    """ Global Norm-Aware Pooling block
    """
    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature
'''

'''
class GDC(Module):
    """ Global Depthwise Convolution block
    """
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c,
                                     groups=in_c,
                                     kernel=(7, 7),
                                     stride=(1, 1),
                                     padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x
'''

'''
class SEModule(Module):
    """ SE block
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x
'''

'''
class BasicBlockIR(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut
'''

'''
class BottleneckIR(Module):
    """ BasicBlock with bottleneck for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut
'''

'''
class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))
'''

'''
class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))
'''


# class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
#     '''A named tuple describing a ResNet block.'''

'''
def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]
'''

'''
def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]

    return blocks
'''


'''
class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], \
            "mode should be ir or ir_se"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == 'ir':
                unit_module = BasicBlockIR
            elif mode == 'ir_se':
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == 'ir':
                unit_module = BottleneckIR
            elif mode == 'ir_se':
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(output_channel),
                                           Dropout(0.4), Flatten(),
                                           Linear(output_channel * 7 * 7, 512),
                                           BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        initialize_weights(self.modules())


    def forward(self, x, return_intermediate=False, return_style=[], return_spatial=[]):

        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)
        styles = []
        spatials = []
        intermediate = None

        for idx, module in enumerate(self.body):
            x = module(x)
            if idx in return_style:
                B,C,H,W = x.shape
                mean = x.view(B,C,-1).mean(-1, keepdim=True)
                std = x.view(B,C,-1).std(-1, keepdim=True)
                style = torch.cat([mean, std], dim=-1)
                styles.append(style)
            if idx in return_spatial:
                spatials.append(x)

        if return_intermediate:
            intermediate = x

        x = self.output_layer(x)
        norm = torch.clip(torch.norm(x, 2, 1, True), 1e-5)
        output = torch.div(x, norm)

        if return_intermediate:
            return output, norm, intermediate

        if return_style and return_spatial:
            return output, norm, styles, spatials

        if return_spatial:
            return output, norm, spatials

        if return_style:
            return output, norm, styles

        return output, norm
'''

'''
def IR_18(input_size):
    """ Constructs a ir-18 model.
    """
    model = Backbone(input_size, 18, 'ir')

    return model
'''

'''
def IR_34(input_size):
    """ Constructs a ir-34 model.
    """
    model = Backbone(input_size, 34, 'ir')

    return model
'''

'''
def IR_50(input_size):
    """ Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model
'''

'''
def IR_101(input_size):
    """ Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model
'''

'''
def IR_152(input_size):
    """ Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model
'''

'''
def IR_200(input_size):
    """ Constructs a ir-200 model.
    """
    model = Backbone(input_size, 200, 'ir')

    return model
'''

'''
def IR_SE_50(input_size):
    """ Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model
'''

'''
def IR_SE_101(input_size):
    """ Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model
'''

'''
def IR_SE_152(input_size):
    """ Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model
'''

'''
def IR_SE_200(input_size):
    """ Constructs a ir_se-200 model.
    """
    model = Backbone(input_size, 200, 'ir_se')

    return model
'''




'''
############# mobilenet

class Conv_block(Module):
    """ Convolution block with no-linear activation layer
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x
'''

'''
class Depth_Wise(Module):
    """ Depthwise block
    """
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, residual=False):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, groups, (1, 1), (1, 1), (0, 0))
        self.conv_dw = Conv_block(groups, groups, kernel, stride, padding, groups=groups)
        self.project = LinearBlock(groups, out_c, (1, 1), (1, 1), (0, 0))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output
'''

'''
class Residual(Module):
    """ Residual block
    """
    def __init__(self, channel, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(channel, channel,
                                      kernel=kernel,
                                      stride=stride,
                                      padding=padding,
                                      groups=groups,
                                      residual=True))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)
'''
