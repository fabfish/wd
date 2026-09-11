"""
Models for CIFAR-100 classification: ResNet-18, ResNet-50, and VGG-16.
Plus a plain MLP for the small-dataset (CIFAR-10/MNIST) arm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic Block for ResNet-18"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck Block for ResNet-50/101/152"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture for CIFAR-100"""

    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes=100):
    """Returns a ResNet-18 model for CIFAR-100"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet50(num_classes=100):
    """Returns a ResNet-50 model for CIFAR-100"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """VGG architecture adapted for CIFAR (32x32 input, smaller FC layers)"""

    def __init__(self, vgg_name, num_classes=100):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg_vgg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def vgg16(num_classes=100):
    """Returns a VGG-16 model for CIFAR-100"""
    return VGG('VGG16', num_classes=num_classes)


class MLP(nn.Module):
    """
    Plain ReLU MLP for CIFAR-10 / MNIST, mirroring mlp_wd.mlp_core.models.MLP
    so the E12 small-dataset arm stays comparable with the established MLP
    experiments. `use_bn` inserts BatchNorm1d after every hidden Linear,
    bringing the network into the (approximately) scale-invariant regime.

    num_layers=3 means input -> hidden -> hidden -> output.
    """

    def __init__(self, in_features=3072, hidden_dim=512, num_layers=3,
                 num_classes=10, use_bn=False):
        super(MLP, self).__init__()
        if num_layers < 2:
            raise ValueError('num_layers must be >= 2')
        dims = [in_features] + [hidden_dim] * (num_layers - 1) + [num_classes]
        layers = [nn.Flatten()]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


MLP_IN_FEATURES = {'cifar10': 3 * 32 * 32, 'mnist': 28 * 28}


def mlp(dataset='cifar10', num_classes=10, hidden_dim=512, num_layers=3,
        use_bn=False):
    """MLP factory keyed by dataset (CIFAR-10: 3072, MNIST: 784 inputs)."""
    if dataset not in MLP_IN_FEATURES:
        raise ValueError(f'Unknown dataset for mlp: {dataset}. '
                         f'Available: {list(MLP_IN_FEATURES)}')
    return MLP(in_features=MLP_IN_FEATURES[dataset], hidden_dim=hidden_dim,
               num_layers=num_layers, num_classes=num_classes, use_bn=use_bn)


def get_model(model_name, num_classes=100, dataset='cifar100'):
    """Factory function to get model by name."""
    if model_name == 'mlp':
        return mlp(dataset=dataset, num_classes=num_classes)
    if model_name == 'mlp_bn':
        # E12 extension: scale-invariant MLP (BN after hidden Linears) to test
        # whether the SGD-phase counter-example disappears with BN.
        return mlp(dataset=dataset, num_classes=num_classes, use_bn=True)
    models = {
        'resnet18': resnet18,
        'resnet50': resnet50,
        'vgg16': vgg16,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    return models[model_name](num_classes=num_classes)
