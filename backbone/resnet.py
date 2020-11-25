import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ('ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)  # why bias


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, data):
        raw_data = data
        out = self.conv1(data)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            raw_data = self.downsample(raw_data)
        out += raw_data
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=True):
        '''

        Args:
            block:
            layers:  for example resnet18 [2, 2, 2, 2]
            zero_init_residual:
        '''
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 304
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 152
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                else:
                    pass

    def _make_layer(self, block, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, out_planes * block.expansion, stride),
                nn.BatchNorm2d(out_planes * block.expansion)
            )

        layers = []

        layers.append(block(self.in_planes, out_planes, stride, downsample))
        self.in_planes = out_planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        C_1 = self.conv1(x)
        C_1 = self.bn1(C_1)
        C_1 = self.relu(C_1)
        C_1 = self.maxpool(C_1)

        C_2 = self.layer1(C_1)
        C_3 = self.layer2(C_2)
        C_4 = self.layer3(C_3)
        C_5 = self.layer4(C_4)

        return C_5


def resnet18(pretrained=False, hr_pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        hr_pretrained(bool: if True, returna model pre-trained on high resolution dataset
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        if hr_pretrained:
            print('Loading the high resolution pretrained model ...')
            model.load_state_dict(torch.load("backbone/weights/resnet18_hr_10.pth"), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model
