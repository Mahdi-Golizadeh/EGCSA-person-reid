##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

from configs import *
import torch
import sys
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from torch import nn
import logging
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
sys.path.append('.')

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load("resnet50-0676ba61.pth")
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}



class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, channel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck_(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False, drop_connection_rate=0.0):
        super(Bottleneck_, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        self.drop_connection_rate = drop_connection_rate

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix > 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_connection_rate > 0:
            out = drop_connect(out, p=self.drop_connection_rate, training=self.training) + residual
        else:
            out += residual
        out = self.relu(out)

        return out

class ResNet_(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        self.global_drop_connect_rate = 0.0

        super(ResNet_, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob, drop_connection_rate=self.global_drop_connect_rate * 0.5)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob, drop_connection_rate=self.global_drop_connect_rate)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob, drop_connection_rate=self.global_drop_connect_rate * 0.5)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob, drop_connection_rate=self.global_drop_connect_rate)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob, drop_connection_rate=self.global_drop_connect_rate * 0.5)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob, drop_connection_rate=self.global_drop_connect_rate)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        import pdb
        pdb.set_trace()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True, drop_connection_rate=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma, drop_connection_rate=drop_connection_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def get_all_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4, x3, x2

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet_(Bottleneck_, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model

def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet_(Bottleneck_, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

def resnest200(pretrained=False, from_moco=False, root='~/.encoding/models', **kwargs):
    model = ResNet_(Bottleneck_, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))

    if from_moco:
        # path = '/home/rym/workspace/fgvc7/moco_pretrained_resnest200.pth.tar'
        path = '/home/cgy/Works/fgvc7-master/logs/checkpoint_0059.pth.tar'
        sd = torch.load(path,map_location='cpu')['state_dict']
        new_sd = {}
        model.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )
        for key in sd.keys():
            if 'module.encoder_k' in key:
                new_sd[key[17:]] = sd[key]
        model.load_state_dict(new_sd)
        model.fc = nn.Sequential()
        print('===> Successfully loaded ResNeSt-200 MOCO pretrained model')

    return model

def resnest269(pretrained=False,from_moco=False, root='~/.encoding/models', **kwargs):
    model = ResNet_(Bottleneck_, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
        
    if from_moco:
        # path = '/home/rym/workspace/fgvc7/moco_pretrained_resnest200.pth.tar'
        path = '/home/cgy/Works/fgvc7-master/logs/moco_pretrained_resnest269.pth.tar'
        sd = torch.load(path,map_location='cpu')['state_dict']
        new_sd = {}
        model.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )
        for key in sd.keys():
            if 'module.encoder_k' in key:
                new_sd[key[17:]] = sd[key]
        model.load_state_dict(new_sd)
        model.fc = nn.Sequential()
        print('===> Successfully loaded ResNeSt-269 MOCO pretrained model')
    return model


EPSILON = 1e-12


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class GAT(torch.nn.Module):
    def __init__(self, channel):
        super(GAT, self).__init__()
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k = t if t%2 else t+1
        self.avg_ch = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.max_ch = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.avg_sp = torch.nn.AdaptiveAvgPool1d(1)
        self.max_sp = torch.nn.AdaptiveMaxPool1d(1)
        self.conv_ch_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size= k, padding= "same", device= MODEL_DEVICE)
        self.conv_sp_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size= k, padding= "same", device= MODEL_DEVICE)
        self.conv_ch_2 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size= k, padding= "same", device= MODEL_DEVICE)
        self.conv_sp_2 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size= k + 2, padding= "same", device= MODEL_DEVICE)
        self.elu1 = torch.nn.ELU()
        self.elu2 = torch.nn.ELU()
        self.sig1 = torch.nn.Sigmoid()
        self.sig2 = torch.nn.Sigmoid()
        # self.bn = torch.nn.BatchNorm2d(channel)
    def forward(self, x):
        # retrieve the size of the input
        b, c, h, w = x.size()
        # channel-wise average pooling
        x11 = self.avg_ch(x)
        # channel-wise max pooling
        x12 = self.max_ch(x)
        # spatial average pooling
        x21 = self.avg_sp(x.permute(-4, -1, -2, -3).reshape(b, h*w, c)).reshape(b, h, w, -1)
        # spatial max pooling
        x22 = self.max_sp(x.permute(-4, -1, -2, -3).reshape(b, h*w, c)).reshape(b, h, w, -1)
        # concating channel features
        x11 = torch.cat([x11, x12], dim= -1)
        # concating spatial features
        x21 = torch.cat([x21, x22], dim= -1)
        x11 = self.conv_ch_1(x11.permute(-2, -1, -4, -3)).permute(-2, -1, -4, -3)
        x21 = self.conv_sp_1(x21.permute(-4, -1, -3, -2))
        out = x11  + self.elu2(x21) * x
        out1 = self.conv_ch_2(out)
        out2 = self.conv_sp_2(out)
        return self.sig1(out1) * x + out2
        
# class GAT(nn.Module):
#     def __init__(self, channel):
#         super(GAT, self).__init__()
#         t = int(abs(math.log(channel, 2) + 1) / 2)
#         k = t if t%2 else t+1
#         self.avg_ch = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.max_ch = torch.nn.AdaptiveMaxPool2d((1, 1))
#         self.avg_sp = torch.nn.AdaptiveAvgPool1d(1)
#         self.max_sp = torch.nn.AdaptiveMaxPool1d(1)
#         self.conv_ch_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding="same", device=MODEL_DEVICE)
#         self.conv_sp_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding="same", device=MODEL_DEVICE)
#         self.conv_ch_2 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=k, padding="same", device=MODEL_DEVICE)
#         self.conv_sp_2 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=k + 2, padding="same", device=MODEL_DEVICE)
#         self.elu1 = torch.nn.ELU()
#         self.elu2 = torch.nn.ELU()
#         self.sig1 = torch.nn.Sigmoid()
#         self.sig2 = torch.nn.Sigmoid()
#         self.self_attn = nn.MultiheadAttention(embed_dim=channel, num_heads=1)  # Self-attention mechanism
#         # self.bn = torch.nn.BatchNorm2d(channel)

#     def forward(self, x):
#         # retrieve the size of the input
#         b, c, h, w = x.size()
#         # channel-wise average pooling
#         x11 = self.avg_ch(x)
#         # channel-wise max pooling
#         x12 = self.max_ch(x)
#         # spatial average pooling
#         x21 = self.avg_sp(x.permute(-4, -1, -2, -3).reshape(b, h*w, c)).reshape(b, h, w, -1)
#         # spatial max pooling
#         x22 = self.max_sp(x.permute(-4, -1, -2, -3).reshape(b, h*w, c)).reshape(b, h, w, -1)
#         # concating channel features
#         x11 = torch.cat([x11, x12], dim=-1)
#         # concating spatial features
#         x21 = torch.cat([x21, x22], dim=-1)
#         x11 = self.conv_ch_1(x11.permute(-2, -1, -4, -3)).permute(-2, -1, -4, -3)
#         x21 = self.conv_sp_1(x21.permute(-4, -1, -3, -2))
#         out = x11 + self.elu2(x21) * x
        
#         # Apply self-attention mechanism
#         out = out.permute(2, 0, 1, 3).reshape(w * h, b, -1)
#         out, _ = self.self_attn(out, out, out)
#         out = out.reshape(b, c, h, w).permute(1, 0, 2, 3)
        
#         out1 = self.conv_ch_2(out)
#         out2 = self.conv_sp_2(out)
#         return self.sig1(out1) * x + out2


class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path,):
        super(Baseline, self).__init__()
        print(f"Training started")
        self.base = ResNet(last_stride= last_stride)
        

        self.base.load_param(model_path)
        self.base_1 = nn.Sequential(*list(self.base.children())[0:3])
        self.base_2 = nn.Sequential(*list(self.base.children())[3:4])
        self.base_3 = nn.Sequential(*list(self.base.children())[4:5])
        self.base_4 = nn.Sequential(*list(self.base.children())[5:6])
        self.base_5 = nn.Sequential(*list(self.base.children())[6:])

        self.att1 = GAT(64)
        self.att2 = GAT(256)
        self.att3 = GAT(512)
        self.att4 = GAT(1024)
        self.att5 = GAT(2048)

        self.BN1 = BN2d(64)
        self.BN2 = BN2d(256)
        self.BN3 = BN2d(512)
        self.BN4 = BN2d(1024)
        self.BN5 = BN2d(2048)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):

        x = self.base_1(x)
        x=x+self.att1(x).expand_as(x)
        x = self.BN1(x)

        x = self.base_2(x)
        x=x+self.att2(x).expand_as(x)
        x = self.BN2(x)

        x = self.base_3(x)
        x=x+self.att3(x).expand_as(x)
        x = self.BN3(x)

        x = self.base_4(x)
        x=x+self.att4(x).expand_as(x)
        x = self.BN4(x)

        x = self.base_5(x)
        x=x+self.att5(x).expand_as(x)
        x = self.BN5(x)

        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            
            return feat

            # return self.classifier(feat)

def build_model(num_classes):
    if MODEL_NAME == 'resnet50':
        model = Baseline(num_classes, MODEL_LAST_STRIDE, MODEL_PRETRAIN_PATH)
        return model
    else:
        raise RuntimeError("'{}' is not available".format(MODEL_NAME))
