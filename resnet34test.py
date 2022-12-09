import torchvision
import torch
from torch import nn

resnet34=torchvision.models.resnet34(pretrained=True)
# torch.reshape(100,1,4096,3);#100个 txt文件，进网络，输出 100* （分类个数）
resnet34.layer4[2].add_module('add_linear',nn.Linear(512,10))
print(resnet34)
