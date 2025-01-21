import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

"""
Credit: https://github.com/byyx666/ArchCraft/tree/main
"""


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class BasicBlockRes(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockRes, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if in_planes != planes * self.expansion or stride != 1:
            self.shortcut = DownsampleB(in_planes, planes * self.expansion, stride)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, depth, width, pool, double, dataset=None):
        super().__init__()
        self.layers = nn.ModuleList()

        pool_code = copy.deepcopy(pool)
        double_code = copy.deepcopy(double)
        self.pool_code = pool_code
        # self.first_conv = BasicBlock(3, width, 1)

        self.inplanes = width
        if 'cifar' in dataset or 'tiny' in dataset:
            self.first_conv = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True))
        elif 'imagenet' in dataset:
            self.first_conv = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        for i in range(depth):
            last_width = width
            while i in double_code:
                double_code.remove(i)
                width *= 2

            self.layers.append(BasicBlockRes(last_width, width, 1))

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.out_dim = width

        self.linear = nn.Linear(width, 1)

    def forward(self, x):
        h = self.first_conv(x)
        p_code = copy.deepcopy(self.pool_code)
        for i, layer in enumerate(self.layers):
            while i in p_code:
                p_code.remove(i)
                h = F.max_pool2d(h, 2)
            h = layer(h)
        out = h
        out = self.global_pooling(out)
        features = torch.flatten(out, 1)
        out = self.linear(features)

        return out


def arch_craft(code, dataset=None):
    args = {'depth': code[0], 'width': code[1], 'pool': code[2], 'double': code[3]}
    depth = int(args['depth'])
    width = int(args['width'])
    pool = [int(x) for x in args['pool']]
    double = [int(x) for x in args['double']]
    return Net(depth, width, copy.deepcopy(pool), copy.deepcopy(double), dataset)
