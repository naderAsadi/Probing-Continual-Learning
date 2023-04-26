import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
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
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
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
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, nf, input_size, in_channel=3, zero_init_residual=False
    ):
        super(ResNet, self).__init__()

        self.in_planes = nf
        self.input_size = input_size

        # hardcoded for now
        self.last_hid = nf * 8 * block.expansion
        # self.last_hid = last_hid * (input_size[-1] // 2 // 2 // 2 // 4) ** 2

        self.conv1 = nn.Conv2d(
            in_channel, nf, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nf, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x, layer):
        if layer < 1 or layer > 4:
            layer = 4
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        out = F.relu(self.bn1(self.conv1(x)))
        for lyr in layers[:layer]:
            out = lyr(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    "resnet18": [resnet18, 512],
    "resnet34": [resnet34, 512],
    "resnet50": [resnet50, 2048],
    "resnet101": [resnet101, 2048],
}

feature_dims = {
    "resnet18": {"1": 128, "2": 256, "3": 512, "4": 512},
    "resnet34": {"1": 128, "2": 256, "3": 512, "4": 512},
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(
        self,
        name="resnet50",
        head="mlp",
        nf=64,
        input_size=(3, 32, 32),
        feat_dim=128,
        hidden_dim=256,
        batch_norm=False,
        num_layers=2,
    ):
        super(SupConResNet, self).__init__()
        model_fun, _ = model_dict[name]
        self.encoder = model_fun(nf=nf, input_size=input_size)

        if head == "linear":
            self.head = nn.Linear(self.encoder.last_hid, feat_dim)
        elif head == "mlp":
            self.head = ProjectionMLP(
                self.encoder.last_hid, hidden_dim, feat_dim, batch_norm, num_layers
            )
        else:
            raise NotImplementedError("head not supported: {}".format(head))

    def return_hidden(self, x, layer=-1):
        return self.encoder.return_hidden(x, layer)

    def forward_classifier(self, x, task=None):
        feat = self.head(x)
        return feat

    def forward(self, x, task=None):
        feat = self.head(self.encoder(x))
        # feat = F.normalize(feat, dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""

    def __init__(
        self,
        name="resnet50",
        head="linear",
        nf=64,
        input_size=(3, 32, 32),
        num_classes=10,
    ):
        super(SupCEResNet, self).__init__()
        model_fun, _ = model_dict[name]
        self.encoder = model_fun(nf=nf, input_size=input_size)

        if head == "linear":
            self.head = nn.Linear(self.encoder.last_hid, num_classes)
        elif head == "distlinear":
            self.head = distLinear(self.encoder.last_hid, num_classes)
        else:
            raise NotImplementedError(f"head not supported: {head}")

    def return_hidden(self, x, layer=-1):
        return self.encoder.return_hidden(x, layer)

    def forward_classifier(self, x, task=None):
        return self.head(x)

    def forward(self, x, task=None):
        return self.head(self.encoder(x))


class MultiHeadResNet(nn.Module):
    """encoder + classifier"""

    def __init__(
        self,
        name="resnet50",
        nf=64,
        input_size=(3, 32, 32),
        n_classes_per_head=10,
        n_heads=10,
    ):
        super(MultiHeadResNet, self).__init__()
        model_fun, _ = model_dict[name]
        self.encoder = model_fun(nf=nf, input_size=input_size)

        self.heads = self._make_layers(
            self.encoder.last_hid, n_classes_per_head, n_heads
        )

    def _make_layers(self, dim_in, n_classes, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, n_classes))
        return nn.ModuleList(layers)

    def return_hidden(self, x, layer=-1):
        return self.encoder.return_hidden(x, layer)

    def forward_classifier(self, x, task):
        return self.heads[task](x)

    def forward(self, x, task):
        return self.heads[task](self.encoder(x))


# Classifiers
# -----------------------------------------------------------------------------------


def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(
        self, name="resnet18", feat_dim=model_dict["resnet18"], num_classes=10
    ):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class distLinear(nn.Module):
    def __init__(self, indim, outdim, weight=None):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        if weight is not None:
            self.L.weight.data = Variable(weight)

        self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)

        L_norm = (
            torch.norm(self.L.weight, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.L.weight.data)
        )
        L_normalized = self.L.weight.div(L_norm + 0.00001)

        cos_dist = torch.mm(x_normalized, L_normalized.transpose(0, 1))

        scores = self.scale_factor * (cos_dist)

        return scores


def add_linear(dim_in, dim_out, batch_norm, relu):
    layers = []
    layers.append(nn.Linear(dim_in, dim_out))
    if batch_norm:
        layers.append(nn.BatchNorm1d(dim_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ProjectionMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, feat_dim, batch_norm, num_layers):
        super(ProjectionMLP, self).__init__()

        self.layers = self._make_layers(
            dim_in, hidden_dim, feat_dim, batch_norm, num_layers
        )

    def _make_layers(self, dim_in, hidden_dim, feat_dim, batch_norm, num_layers):
        layers = []
        layers.append(add_linear(dim_in, hidden_dim, batch_norm=batch_norm, relu=True))

        for _ in range(num_layers - 2):
            layers.append(
                add_linear(hidden_dim, hidden_dim, batch_norm=batch_norm, relu=True)
            )

        layers.append(add_linear(hidden_dim, feat_dim, batch_norm=False, relu=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PredictionMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, out_dim):
        super(PredictionMLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)
