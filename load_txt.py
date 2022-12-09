import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch import nn, tensor
from torch.nn import Conv2d, MaxPool2d
import torchvision
from torch.utils.tensorboard import SummaryWriter


class ResBlock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(ResBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=(3,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut=nn.Sequential()
        if inchannel!=outchannel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=(3,1),stride=(2,1)),
                nn.BatchNorm2d(outchannel)
        )
    def forward(self,x):
        pad = nn.ZeroPad2d(padding=(0, 0, 1, 1))  # 左右不填充，上下各填充1
        x = pad(x)  # 对x先增加padding
        out=self.left(x)#left部分做卷积
        out+=self.shortcut(x)#short部分降维
        out=torch.nn.functional.relu(out)
        return out

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.txt_path=os.listdir(self.path)#得到的列表是当前文件夹下面所有样本的文件名字
    def __getitem__(self,idx):
        txt_name=self.txt_path[idx]
        txt_name_path=os.path.join(self.root_dir,self.label_dir,txt_name)#拼接根目录，分类名称，txt文件名字
        txt_content=pd.read_csv(txt_name_path,sep=' ', keep_default_na=False, header=None, encoding="utf-8")
        txt_array=np.array(txt_content)#dataframe类型转换成tensor类型
        txt_tensor=torch.tensor(txt_array)#用torch.reshape重塑维度 -》1×4096×3
        txt_reshape=torch.reshape(txt_tensor,(1,4096,3))
        target=0 if self.label_dir=='danyin_data'or self.label_dir=='danyin_test_data' else 1#单音0多音1
        label_array=np.array(target)
        label_tensor=torch.tensor(label_array,dtype=torch.long)
        return txt_name_path,txt_reshape,label_tensor
    def __len__(self):
        return len(self.txt_path)

root_dir="D:\pytorch_project\dataset"
danyin_label_dir="danyin_data"
duoyin_label_dir="duoyin_data"
danyin_datasets=MyData(root_dir,danyin_label_dir)#实例化对象，并且把参数传给init(构造函数）
duoyin_datasets=MyData(root_dir,duoyin_label_dir)

danyin_test_label_dir="danyin_test_data"
duoyin_test_label_dir="duoyin_test_data"
danyin_test_datasets=MyData(root_dir,danyin_test_label_dir)
duoyin_test_datasets=MyData(root_dir,duoyin_test_label_dir)
# txt_name_path,txt1,label=danyin_datasets.__getitem__(1)
# print(txt1,"维度：",txt1.shape,"分类：",label,"数据集大小：",len(danyin_datasets),txt_name_path)

train_datasets=danyin_datasets+duoyin_datasets
test_datasets=danyin_test_datasets+duoyin_test_datasets
train_data_size=len(train_datasets)
test_data_size=len(test_datasets)
print("训练集大小：{}".format(train_data_size))
print("测试集大小：{}".format(test_data_size))

# txt_name_path,content,label=train_datasets[0]
# print(content)
# print(content.shape,label)

#一次性加载4个样本
train_loader=DataLoader(dataset=train_datasets,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
test_loader=DataLoader(dataset=test_datasets,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
#看一下加载数据到网络时候的数据维度
# for data in test_loader:
#     name_path,txt,target=data
#     print(txt.shape)
#     print(target)


class mymodule(nn.Module):
    def __init__(self):
        super(mymodule, self).__init__()
        self.inchannel=16
        self.conv1=Conv2d(1,16,kernel_size=(3,1),stride=1,padding=0)#
        self.maxpool1=MaxPool2d(kernel_size=(2,1))#不设置stride,stride默认为kernelsize
        self.conv2=Conv2d(16,16,kernel_size=(3,1),stride=1,padding=0)
        self.maxpool2=MaxPool2d(kernel_size=(2,1))
        #resnet
        # self.layer1=self.make_layer(ResBlock,16)
        self.layer2=self.make_layer(ResBlock,32)
        self.layer3=self.make_layer(ResBlock,64)
        self.layer4=self.make_layer(ResBlock,128)
        self.layer5=self.make_layer(ResBlock,256)
        self.gap=nn.AvgPool2d(kernel_size=(64,3))
        self.flatten=nn.Flatten()
        # self.linear1=nn.Linear(49152,64)
        # self.linear2=nn.Linear(64,2)
        self.linear1=nn.Linear(256,64)
        self.linear2=nn.Linear(64,16)
        self.linear3=nn.Linear(16,2)
        #resblock
    def make_layer(self,block,channels):#比如resnet18,一个layer包含多个resblock，这里是5个resblock，
        #为了和输入输出维度匹配，这里用5个layers，每个layers只包含一个resnet，这样每经过一个layers就
        layers=[]
        layers.append(block(self.inchannel,channels))#第一次 self.inchannel=16，channels=32
        self.inchannel=channels
        return  nn.Sequential(*layers)

    def forward(self,x):
        pad=nn.ZeroPad2d(padding=(0,0,1,1))#左右不填充，上下各填充1
        x=pad(x)#先增加padding
        x = x.to(torch.float32)
        # print(x,x.shape)# [4,1,4098,3]
        x=self.conv1(x)#经过卷积层1
        # print(x.shape)#[ 4, 16, 4096,3]
        b1 = nn.BatchNorm2d(16)
        r1 = nn.ReLU(inplace=True)
        x = b1(x)
        x = r1(x)
        x=self.maxpool1(x)
        # print(x.shape)#[4, 16, 2048, 3]
        x=pad(x)#再次上下填充1行
        x=self.conv2(x)
        # print(x.shape)#[4, 16, 2048, 3]
        b2 = nn.BatchNorm2d(16)
        r2 = nn.ReLU(inplace=True)

        x = b2(x)
        x = r2(x)
        x=self.maxpool2(x)
        # print(x.shape)#[4, 16, 1024, 3]
        #resblock 层
        # x=self.layer1(x)
        # print(x.shape)
        x=self.layer2(x)
        # print(x.shape)#[4, 32, 512, 3]
        x=self.layer3(x)
        # print(x.shape)#[4, 64, 256, 3]
        x=self.layer4(x)
        # print(x.shape)#[4, 128, 128, 3]
        x=self.layer5(x)
        # print(x.shape)#[4, 256, 64, 3]
        x=self.gap(x)
        # print(x.shape)#[4, 256, 1, 1]
        x=self.flatten(x)
        # print(x.shape)#[4, 256]
        x=self.linear1(x)
        # print(x.shape)#[4, 64]
        x=self.linear2(x)
        # print(x.shape)#[4, 16]
        x=self.linear3(x)
        # print(x.shape)#[4, 2]
        return x


# print(# self.path = "D:\\pytorch_project\\dataset\\danyin_data"
#         # self.filelist = os.listdir(self.path)
#         #
#         # dfList = []
#         # for i in filelist:
#         #     print(i)
#         #     filepath = os.path.join(path, i)
#         #     df = pd.read_csv(filepath, sep=' ', keep_default_na=False, header=0, encoding="utf-8")
#         #     dfList.append(df)
#         # dfAll = pd.concat(dfList)
#         # print(dfAll.shape)
#         # print(dfAll)len(filelist))

#每次读取10个txt文件，每个txt文件是一种分类，10个为一个batch，输网络，输出分类结果
# input=torch.tensor(dfAll)
# input=torch.reshape(input,(-1,1,4096,3))
# print(input.shape)

# torchvision.models.resnet34(pretrained=False)
# torch.reshape(100,1,4096,3);#100个 txt文件，进网络，输出 100* （分类个数）
#
# class myModel(nn.Module):
#     def __init__(self):
#         super(myModel, self).__init__()
#         self.conv1=Conv2d(1,16,kernel_size=(3,1),stride=1,padding='same')
#         self.maxpo1=MaxPool2d(kernel_size=(2,1),stride=1)
#         self.conv2=Conv2d(16,16,kernel_size=(2,3),strid=1,padding=)
#         self.maxpo2=MaxPool2d(kernel_size=(2,1),stride=1)

mymodule1=mymodule()
#损失函数
loss_fnc=nn.CrossEntropyLoss()
learning_rate=0.0001
optimizer=torch.optim.SGD(mymodule1.parameters(),lr=learning_rate)
#设置训练网络的一些参数
total_train_step=0#记录训练的参数
total_test_step=0#测试次数
#训练总轮数
epoch=100
#添加tensorboard
writer=SummaryWriter("../logs_train")


for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    mymodule1.train()
    for data in train_loader:
        txt_name_path,txt,target=data
        # print(txt_name_path,txt,target)
        outputs=mymodule1(txt)
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
            txt_name_path, txt, target = data
            outputs = mymodule1(txt)
            loss = loss_fnc(outputs, target)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(outputs.argmax(1)==target).sum()# sum是统计true的个数共有多少
            total_accuracy=total_accuracy+accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accuracy/train_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/train_data_size,total_test_step)
    total_test_step=total_test_step+1
writer.close()
