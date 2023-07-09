import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from dataclasses import dataclass

class Residual_Block(nn.Module):
    def __init__(
            self,
            in_channels = 0,
            out_channels = 0,
            stride = 1,
            downsample = None,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding = 1, kernel_size = 3, stride = stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, padding = 'same', kernel_size = 3, stride = 1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.out_channels = out_channels
        self.ReLU = nn.ReLU(False)
    
    def forward(self, X):
        residual = X
        out = self.conv1(X)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(X)

        out += residual
        out = self.ReLU(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Residual_Block,
        layers = None,
        input_size = [3, 224, 224],
        output_size = [68, 2],
    ):
        super().__init__()
        # print(block)
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size[0], 64, padding = 3, kernel_size = 7, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(False),
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer0 = self._make_layer(block, 64, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 256, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1) ### notice this line

        self.adapool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(512, self.output_size[0] * self.output_size[1])
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride = 1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride = stride, downsample = downsample))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride = 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):       
        x = self.conv1(x) ##224 -> 112
        x = self.maxpool(x) ##112 -> 56

        x = self.layer0(x) ##56 -> 56
        x = self.layer1(x) ##56 -> 28
        x = self.layer2(x) ##28 -> 14
        x = self.layer3(x) ##14 -> 7

        # # print(x.size())
        x = self.avgpool(x) ##7 -> 1
        # # print(x.size())
        # # x = x.view(x.size(0), -1)
        x = x.squeeze()
        # # print(x.size())

        # x = nn.Linear(512, self.output_size[0] * self.output_size[1], device = next(self.parameters()).device)(x) ##notice this line?
        
        # print(x.size())
        # x = self.adapool(x) ##7 -> 1
        # print(x.size())
        # x = x.squeeze()
        x = self.fc(x)

        # x = nn.Linear(x.size(1), self.output_size[0] * self.output_size[1], device = torch.device("cuda"))(x)

        return x.reshape(x.size(0), self.output_size[0], self.output_size[1])

if __name__ == "__main__":
    def test():
        print(Residual_Block)
        print(ResNet)

        print(Residual_Block(3, 64, 1, downsample = nn.Conv2d(3, 64, kernel_size=1, stride=1))(torch.randn((5, 3, 69, 69))).size())
        model = ResNet(Residual_Block, layers=[2, 2, 2, 2])
        print(model)
        params = list(model.parameters())
        print(params[0].size())
        intput = torch.randn((5, 3, 224, 224))
        output = model(intput)
        print(output.size())

    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    # from src.models.components.simple_resnet import SimpleResnet
    
    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    output_path = path / "outputs"
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    print("paths", path, config_path, output_path)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="my_resnet")
    def main(cfg: DictConfig):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = hydra.utils.instantiate(cfg.get('net'))
        net.to(device)
        print(net)
        print(device)
        input = torch.randn((5, 3, 224, 224), device=device)

        output = net(input)
        print(output.size())

    # # target output size of 5x7
    # m = nn.AdaptiveAvgPool2d((5, 7))
    # input = torch.randn(3, 64, 8, 9)
    # output = m(input)
    # print(output.size())
    # # target output size of 7x7 (square)
    # m = nn.AdaptiveAvgPool2d(7)
    # input = torch.randn(1, 64, 10, 9)
    # output = m(input)
    # # target output size of 10x7
    # m = nn.AdaptiveAvgPool2d((None, 7))
    # input = torch.randn(1, 64, 10, 9)
    # output = m(input)
    main()
