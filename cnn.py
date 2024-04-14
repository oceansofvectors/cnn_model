from torchvision import models
import torch.nn as nn


class ResNetGray(models.ResNet):
    def __init__(self, num_classes):
        super().__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512, num_classes)
