"""
Some architectures for the DnCNN model.
Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

"""
DnCNN pytorch
"""

# --- Libraries --- #
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from pytorchcv.model_provider import get_model
from torchvision.models import resnet
from collections import OrderedDict


def forward_resnet_conv(net, x, upto: int = 4):
    """
    Forward ResNet only in its convolutional part
    :param net:
    :param x:
    :param upto:
    :return:
    """
    x = net.conv1(x)  # N / 2
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)  # N / 4

    if upto >= 1:
        x = net.layer1(x)  # N / 4
    if upto >= 2:
        x = net.layer2(x)  # N / 8
    if upto >= 3:
        x = net.layer3(x)  # N / 16
    if upto >= 4:
        x = net.layer4(x)  # N / 32
    return x


class Head(nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out


class FCN(nn.Module):
    def __init__(self, base, in_f, out_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, out_f)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str, n_classes: int, pretrained: bool):
        super(EfficientNetGen, self).__init__()

        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained(model)
        else:
            self.efficientnet = EfficientNet.from_name(model)

        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, n_classes)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        # comment the softmax
        # x = F.softmax(x, dim=-1)
        return x


class EfficientNetB0(EfficientNetGen):
    def __init__(self, n_classes: int, pretrained: bool):
        super(EfficientNetB0, self).__init__(model='efficientnet-b0', n_classes=n_classes, pretrained=pretrained)


class EfficientNetB4(EfficientNetGen):
    def __init__(self, n_classes: int, pretrained: bool):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4', n_classes=n_classes, pretrained=pretrained)


"""
EfficientNet with attention
"""

class EfficientNetAutoAtt(EfficientNet):
    def init_att(self, model: str, width: int):
        """
        Initialize attention
        :param model: efficientnet-bx, x \in {0,..,7}
        :param depth: attention width
        :return:
        """
        if model == 'efficientnet-b4':
            self.att_block_idx = 9
            if width == 0:
                self.attconv = nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)
            else:
                attconv_layers = []
                for i in range(width):
                    attconv_layers.append(
                        ('conv{:d}'.format(i), nn.Conv2d(kernel_size=3, padding=1, in_channels=56, out_channels=56)))
                    attconv_layers.append(
                        ('relu{:d}'.format(i), nn.ReLU(inplace=True)))
                attconv_layers.append(('conv_out', nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)))
                self.attconv = nn.Sequential(OrderedDict(attconv_layers))
        else:
            raise ValueError('Model not valid: {}'.format(model))

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:

        # Placeholder
        att = None

        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                break

        return att

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                x = x * att

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

class EfficientNetGenAutoAtt(FeatureExtractor):
    def __init__(self, model: str, width: int, n_classes: int, pretrained: bool):
        super(EfficientNetGenAutoAtt, self).__init__()

        if pretrained:
            self.efficientnet = EfficientNetAutoAtt.from_pretrained(model)
        else:
            self.efficientnet = EfficientNetAutoAtt.from_name(model)

        self.efficientnet.init_att(model, width)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, n_classes)
        # self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet.get_attention(x)


class EfficientNetAutoAttB4(EfficientNetGenAutoAtt):
    def __init__(self, n_classes: int, pretrained: bool):
        super(EfficientNetAutoAttB4, self).__init__(model='efficientnet-b4', width=0, n_classes=n_classes, pretrained=pretrained)


"""
Xception from Kaggle
"""


class XceptionWeiHao(FeatureExtractor):

    def __init__(self, n_classes: int, pretrained: bool):
        super(XceptionWeiHao, self).__init__()

        self.model = get_model("xception", pretrained=pretrained)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove original output layer
        self.model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.model = FCN(self.model, 2048, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.base(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.model.h1(x)

"""
ResNet50 
"""


class ResNet50(FeatureExtractor):
    def __init__(self, n_classes: int, pretrained: bool):
        super(ResNet50, self).__init__()
        self.resnet = resnet.resnet50(pretrained=pretrained)
        self.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)
        del self.resnet.fc

    def features(self, x):
        x = forward_resnet_conv(self.resnet, x)
        x = self.resnet.avgpool(x).flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

"""
DnCNN pytorch
"""


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
                                    padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
