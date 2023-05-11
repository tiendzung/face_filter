import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from dataclasses import dataclass

class conv_block(nn.Module):
  def __init__(
      self,
      num_channels = 0, 
      out_channels = 0,
      stride = 1,
      downsample = None,
  ):
    super().__init__()
    self.conv1 = nn.Conv2d(num_channels, out_channels,  padding = 'same', kernel_size = 3, stride = 1)
    self.batch_norm_1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, padding = 'same', kernel_size = 3, stride = 1)
    self.batch_norm_2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample

  def forward(self, X):
    X = self.conv1(X)
    X = self.batch_norm_1(X)
    X = nn.ReLU(False)(X)

    X = self.conv2(X)
    X = self.batch_norm_2(X)
    X = nn.ReLU(False)(X)

    if self.downsample is not None:
      X = self.downsample(X)

    return X

class ConvNet(nn.Module):
  def __init__(
      self,
      block,
      layers = None,
      input_size = [3, 224, 224],
      output_size = [68, 2],
      ):
    super().__init__()
    print(block)
    self.input_size = input_size
    self.output_size = output_size

    self.layer1 = self._make_layer(block, input_size[0], 64, layers[0], stride=1)
    self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=1)
    self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=1)
    self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=1)
    
    self.classifier = nn.Sequential(
        nn.Linear(14 * 14 * 512, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, self.output_size[0] * self.output_size[1]),
    )

  def _make_layer(self,block, num_channels, out_channels, blocks, stride = 1):
    downsample = None
    if stride != 1 or num_channels != out_channels:
      downsample = nn.MaxPool2d(2, 2)
    
    layers = []
    layers.append(block(num_channels, out_channels, stride, downsample))
    num_channels = out_channels
    for _ in range(1, blocks):
        layers.append(block(num_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, X):
    X = self.layer1(X) # 112
    X = self.layer2(X) # 56
    X = self.layer3(X)  # 28
    X = self.layer4(X) #14

    X = X.view(X.size(0), -1) #N, 12 * 14 * 14

    X = self.classifier(X)
    return X.reshape(X.size(0), self.output_size[0], self.output_size[1])

if __name__ == "__main__":
    print(conv_block)
    print(ConvNet)

    print(conv_block(64, 128)(torch.randn((5, 64, 224,224))).shape)
    net = ConvNet(conv_block, layers = [1, 1 ,1 ,1])
    params = list(net.parameters())
    print(len(params))
    print(params[0].size()) 
    print(net)
    input = torch.randn(16, 3, 224, 224)
    out = net(input)
    print(out.size())
