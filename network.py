import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        out = self.left(x)  # left部分做卷积
        out += x
        out = F.relu(out)
        return out


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 不设置stride,stride默认为kernelsize

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)




        self.layer1 = ResBlock(64, 64)
        self.layer2 = ResBlock(64, 64)

        self.conv4 = nn.Conv2d(64, 16, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm2d(16)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # self.fc1 = nn.Linear(2304, 512)#centercrop后输出时是 16x2304

        self.fc1=nn.Linear(3072,512)#不增加centercrop输出是 16x3072
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.conv1(x)  # 经过卷积层1
        x = F.relu(self.batchnorm1(x))
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        x = self.maxpool3(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = F.relu(self.batchnorm4(x))
        x = self.maxpool4(x)

        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    a = torch.randn(4, 3, 224, 224)
    m = model()
    x = m(a)
