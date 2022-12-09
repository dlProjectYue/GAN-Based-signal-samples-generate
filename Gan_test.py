import os

import numpy as np
from torchvision.datasets import ImageFolder
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
# from network import model
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import  matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.optim as optim

if __name__ == '__main__':
    transform = T.Compose([
            T.Resize([224,224]),
            # T.CenterCrop(224),
            # T.Grayscale(1),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            # T.Normalize(mean=[.5], std=[.5])
        ])
    nc=3
    nz=100
    ngf=64
    ndf=64
    num_epochs=100
    lr1=0.1 #0.0002 0.001是有图案的
    lr2=0.1
    beta1=0.5#adam param
    ngpu=1
    train_path = r'C:\Users\de\Desktop\Gan_train'
    dataset = ImageFolder(train_path, transform=transform)
    print(dataset.class_to_idx)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    real_batch=next(iter(dataloader))
    plt.figure(figsize=(4,8))# batch_size=32，就是4*8展示
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:32],padding=2,normalize=True).cpu(),(1,2,0)))
    def weights_init(m):
        classname=m.__class__.__name__
        if classname.find('Conv') !=-1:
            nn.init.normal_(m.weight.data,0.0,0.02)
        elif classname.find('BatchNorm') !=-1:
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)

    class Generator(nn.Module):
        def __init__(self,ngpu):
            super(Generator,self).__init__()
            self.ngpu=ngpu
            self.main=nn.Sequential(

                    nn.ConvTranspose2d(in_channels=nz,out_channels=ngf*16,kernel_size=7,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(ngf*16),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=ngf*16,out_channels=ngf*8,kernel_size=4,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(ngf*8),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=ngf*8,out_channels=ngf*4,kernel_size=4,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(ngf*4),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=ngf*4,out_channels=ngf*2,kernel_size=4,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(ngf*2),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=ngf*2,out_channels=ngf,kernel_size=4,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=ngf,out_channels=nc,kernel_size=4,stride=2,padding=1,bias=False),
                    nn.Tanh()
            )
        def forward(self,input): # x表示长度为100的噪声
            return self.main(input)

    netG=Generator(ngpu).to(device)
    if(device.type=='cuda') and(ngpu>1):
        netG=nn.DataParallel(netG,list(range(ngpu)))
    netG.apply(weights_init)
    print(netG)
    class Discriminator(nn.Module):
        def __init__(self,ngpu):
            super(Discriminator,self).__init__()
            self.ngpu=ngpu
            self.main=nn.Sequential(  #input (3,224,224)
                nn.Conv2d(nc,ndf,4,2,1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),#output(64,112,112)
                nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),#input (64,112,112)
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2,inplace=True),#output(128,56,56)
                nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),#input(128,56,56)
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU(0.2,inplace=True),#output(256,28,28)
                nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),#input(256,28,28)
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2,inplace=True),#output(512,14,14)
                nn.Conv2d(ndf*8,ndf*16,4,2,1,bias=False),
                nn.BatchNorm2d(ndf*16),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(ndf*16,1,7,1,0,bias=False),#input(512,14,14)
                nn.Sigmoid()#二分类 ，用Sigmoid [0 ,1] output(512,1,1)
            )
        def forward(self,input):
            return self.main(input)
    netD=Discriminator(ngpu).to(device)
    if(device.type=='cuda')and (ngpu>1):
        netD=nn.DataParallel(netD,list(range(ngpu)))
    netD.apply(weights_init)
    print(netD)
    criterion=nn.BCELoss()
    fixed_noise=torch.randn(4,nz,1,1,device=device)
    real_label=1.
    fake_label=0.
    optimizerD=optim.Adam(netD.parameters(),lr=lr1,betas=(beta1,0.999))#分类器
    optimizerG=optim.Adam(netG.parameters(),lr=lr2,betas=(beta1,0.999))#生成器

    img_list=[]
    G_losses=[]
    D_losses=[]
    iters=0
    print("starting Training Loop")
    for lr1 in np.arange (0.1,0.2,0.01):
        for lr2 in np.arange (0.1,0.2,0.01):
            for epoch in range(num_epochs):
                for i,data in enumerate(dataloader,0):
                    netD.zero_grad()
                    real_cpu=data[0].to(device)
                    b_size=real_cpu.size(0)
                    label=torch.full((b_size,),real_label,dtype=torch.float,device=device)
                    # print("真label",label)
                    output=netD(real_cpu).view(-1)
                    # print("discriminator结果",output)
                    errD_real=criterion(output,label)
                    errD_real.backward()
                    D_x=output.mean().item()
                    noise=torch.randn(b_size,nz,1,1,device=device)
                    # print("随机噪声",noise)
                    fake=netG(noise)#generate
                    # print("随机噪声经过生成器",fake)
                    label.fill_(fake_label)
                    # print("假label",label)
                    output=netD(fake.detach()).view(-1)
                    # print("假图片经过discriminator",output)
                    errD_fake=criterion(output,label)
                    errD_fake.backward()
                    D_G_z1=output.mean().item()
                    errD=errD_real+errD_fake
                    optimizerD.step()

                    netG.zero_grad()
                    label.fill_(real_label)
                    output=netD(fake).view(-1)
                    errG=criterion(output,label)
                    errG.backward()
                    D_G_z2=output.mean().item()
                    optimizerG.step()

                    if i%50==0:
                        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                              %(epoch,num_epochs,i,len(dataloader),
                              errD.item(),errG.item(),D_x,D_G_z1,D_G_z2))
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

                    if(iters % 500 == 0 ) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                        with torch.no_grad():
                            fake=netG(fixed_noise).detach().cpu()
                        img_list.append(vutils.make_grid(fake,padding=2,normalize=True))
                    iters+=1
            # real_batch=next(iter(dataloader))
            # plt.figure(figsize=(15,15))
            # plt.subplot(1,2,1)
            # plt.axis("off")
            # plt.title("Real Images")
            # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:32],padding=5,normalize=True).cpu(),(1,2,0)))

            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))
            # plt.show()
            plt.savefig('C:/Users/de/Desktop/深度学习/pic/lr1_{}_lr2_{}.png'.format(lr1,lr2))
