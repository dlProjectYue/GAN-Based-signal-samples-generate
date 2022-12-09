import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch import nn, tensor
from torch.nn import Conv2d, MaxPool2d
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import  Image
from torchvision import transforms


class ResBlock(nn.Module):
    def __init__(self,inchannel,outchannel,strides):
        super(ResBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=strides,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut=nn.Sequential()
        if strides!=1 or inchannel!=outchannel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=strides,padding=1),
                nn.BatchNorm2d(outchannel)
        )
    def forward(self,x):
        # pad = nn.ZeroPad2d(padding=(0, 0, 1, 1))  # 左右不填充，上下各填充1
        # x = pad(x)  # 对x先增加padding
        out=self.left(x)#left部分做卷积
        out+=self.shortcut(x)#short部分降维
        out=torch.nn.functional.relu(out)
        return out





#导入自制数据集
writer=SummaryWriter("img_logs")
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)#得到的列表是当前文件夹下面所有样本的文件名字
    def __getitem__(self,idx):
        img_name=self.img_path[idx]
        img_name_path=os.path.join(self.root_dir,self.label_dir,img_name)#拼接根目录，分类名称，txt文件名字
        img_content=Image.open(img_name_path)
        trans_gray=transforms.Grayscale(num_output_channels=1)
        img_gray=trans_gray(img_content)


        trans_resize=transforms.Resize((224,224))
        img_resize=trans_resize(img_gray)

        tensor_trans=transforms.ToTensor()
        img_tensor=tensor_trans(img_resize)
        # trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # img_norm = trans_norm(img_tensor)
        target=0 if self.label_dir=='danyin_data_jpg'or self.label_dir=='danyin_test_data_jpg' else 1#单音0多音1
        label_array=np.array(target)
        label_tensor=torch.tensor(label_array,dtype=torch.long)
        return img_name_path,img_tensor,label_tensor
    def __len__(self):
        return len(self.img_path)

root_dir="D:\pytorch_project\dataset"
danyin_label_dir="danyin_data_jpg"
duoyin_label_dir="duoyin_data_jpg"
danyin_datasets=MyData(root_dir,danyin_label_dir)#实例化对象，并且把参数传给init(构造函数）
duoyin_datasets=MyData(root_dir,duoyin_label_dir)

danyin_test_label_dir="danyin_test_data_jpg"
duoyin_test_label_dir="duoyin_test_data_jpg"
danyin_test_datasets=MyData(root_dir,danyin_test_label_dir)
duoyin_test_datasets=MyData(root_dir,duoyin_test_label_dir)

img_name_path,img_tensor,label=duoyin_datasets[2]
# print(img_tensor)
# print(img_name_path,img_tensor.shape,label)
# writer.add_image("gray_resize",img_tensor)
train_datasets=danyin_datasets+duoyin_datasets
test_datasets=danyin_test_datasets+duoyin_test_datasets
train_data_size=len(train_datasets)
test_data_size=len(test_datasets)
print("训练集大小：{}".format(train_data_size))
print("测试集大小：{}".format(test_data_size))

train_loader=DataLoader(dataset=train_datasets,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
test_loader=DataLoader(dataset=test_datasets,batch_size=4,shuffle=True,num_workers=0,drop_last=False)

class mymodule(nn.Module):
    def __init__(self):
        super(mymodule, self).__init__()
        self.inchannel=16
        self.conv1=Conv2d(1,16,kernel_size=3,stride=1,padding=1)#
        self.maxpool1=MaxPool2d(kernel_size=2)#不设置stride,stride默认为kernelsize
        self.conv2=Conv2d(16,16,kernel_size=3,stride=1,padding=1)
        self.maxpool2=MaxPool2d(kernel_size=2)
        #resnet
        self.layer1=self.make_layer(ResBlock,3,16,strides=1)
        self.layer2=self.make_layer(ResBlock,3,32,strides=2)
        self.layer3=self.make_layer(ResBlock,3,64,strides=2)

        self.gap=nn.AvgPool2d(kernel_size=(14,14))
        self.flatten=nn.Flatten()
        # self.linear1=nn.Linear(49152,64)
        # self.linear2=nn.Linear(64,2)
        self.linear1=nn.Linear(64,2)
        #resblock
    def make_layer(self,block,block_nums,channels,strides):#比如resnet18,一个layer包含多个resblock，这里是5个resblock，
        #为了和输入输出维度匹配，这里用5个layers，每个layers只包含一个resnet，这样每经过一个layers就
        stride_num=[strides]+[1]*(block_nums-1)
        layers=[]
        for stride in stride_num:
            layers.append(block(self.inchannel,channels,stride))#第一次 self.inchannel=16，channels=32
            self.inchannel=channels
        return  nn.Sequential(*layers)

    def forward(self,x):
        x = x.to(torch.float32)
        # print(x,x.shape)# [4,1,224,224]
        x=self.conv1(x)#经过卷积层1
        # print(x.shape)#[ 4, 16, 224,224]
        b1=nn.BatchNorm2d(16)
        r1=nn.LeakyReLU(0.1)
        x=b1(x)
        x=r1(x)
        x=self.maxpool1(x)
        # print(x.shape)#[4, 16, 112, 112]
        x=self.conv2(x)
        b2 = nn.BatchNorm2d(16)
        r2 = nn.LeakyReLU(0.1)
        x = b2(x)
        x = r2(x)
        # print(x.shape)#[4, 16, 112, 112]
        x=self.maxpool2(x)
        # print(x.shape)#[4, 16, 56, 56]
        #resblock 层
        x=self.layer1(x)
        # print(x.shape)
        x=self.layer2(x)
        # print(x.shape)#[4, 32, 28, 28]
        x=self.layer3(x)
        # print(x.shape)#[4, 64, 14, 14]
        x=self.gap(x)
        # print(x.shape)#[4, 64, 1, 1]
        x=self.flatten(x)
        # print(x.shape)#[4, 64]
        x=self.linear1(x)
        # print(x.shape)#[4, 2]
        return x
mymodule1=mymodule()
#损失函数
loss_fnc=nn.CrossEntropyLoss()
learning_rate=0.0001
optimizer=torch.optim.Adam(mymodule1.parameters(),lr=learning_rate,betas=(0.9,0.99),eps=1e-6,weight_decay=0.0005)
#设置训练网络的一些参数
total_train_step=0#记录训练的参数
total_test_step=0#测试次数
#训练总轮数
epoch=10
#添加tensorboard
writer=SummaryWriter("../logs_train_img")
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    mymodule1.train()
    for data in train_loader:
        img_name_path,img,target=data
        # print(txt_name_path,txt,target)
        outputs=mymodule1(img)
        loss=loss_fnc(outputs,target)

        #优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step=total_train_step+1
        if(total_train_step%100==0):
            print("训练次数：{},loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试集开始

    mymodule1.eval()
    total_accuracy=0
    total_test_loss=0
    with torch.no_grad():
        for data in test_loader:
            img_name_path, img, target = data
            outputs = mymodule1(img)
            loss = loss_fnc(outputs, target)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(torch.argmax(torch.softmax(outputs, dim=1), dim=1)==target).sum()# sum是统计true的个数共有多少
            total_accuracy=total_accuracy+accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accuracy/train_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/train_data_size,total_test_step)
    total_test_step=total_test_step+1


writer.close()