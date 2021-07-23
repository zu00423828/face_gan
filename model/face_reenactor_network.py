from numpy import str0
import torch
import torch.nn as nn
from torch.nn.modules import pooling
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.container import ModuleList
from torch.nn.modules.conv import Conv2d
from torchvision import models, transforms
from torchvision.models.resnet import conv3x3
class Pix2Pix(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Pix2Pix,self).__init__()
        self.out_channel=out_channel
        encoder_layer=[]
        layer=nn.Sequential(nn.Conv2d(in_channel,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.2,inplace=True))
        encoder_layer.append(layer)
        layer_specs=[
            64*2,
            64*4,
            64*8,
            64*8,
            64*8,
            64*8,
            64*8
        ]
        for idx,out_channels in enumerate(layer_specs):
            if idx==0:
                layer=nn.Sequential(nn.Conv2d(64,out_channels,4,stride=2,padding=1),nn.BatchNorm2d(out_channels),nn.LeakyReLU(0.2,inplace=True))
            else:
                layer=nn.Sequential(nn.Conv2d(layer_specs[idx-1],out_channels,4,stride=2,padding=1),nn.BatchNorm2d(out_channels),nn.LeakyReLU(0.2,inplace=True))
            encoder_layer.append(layer)
        self.encoder=ModuleList(encoder_layer)
        layer_specs = [
        64 * 8,  
        64 * 8, 
        64 * 8,
        64 * 8,
        64 * 4,
        64 * 2, 
        64,
        ]
        decoder_layer=[]
        for idx,out_channels in enumerate(layer_specs):
            if idx==0:
                layer=nn.Sequential(nn.ConvTranspose2d(out_channels,out_channels,4,stride=2,padding=1),nn.BatchNorm2d(out_channels),nn.LeakyReLU(0.2,inplace=True))
            else:
                layer=nn.Sequential(nn.ConvTranspose2d(layer_specs[idx-1]*2,out_channels,4,stride=2,padding=1),nn.BatchNorm2d(out_channels),nn.LeakyReLU(0.2,inplace=True))
            decoder_layer.append(layer)
        self.decoder=ModuleList(decoder_layer)
        self.segment_layer=nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(64,1,3,stride=1,padding=1),nn.BatchNorm2d(1),nn.Sigmoid())
        self.last_decoder=nn.Sequential(nn.ConvTranspose2d(128,self.out_channel,4,stride=2,padding=1),nn.Tanh())
    def forward(self,x):
        feature=[]
        out=self.encoder[0](x)
        feature.append(out)
        for layer in self.encoder[1:]:
            out=layer(out)
            feature.append(out)
        for idx,layer in enumerate(self.decoder):
            skip_layer=len(self.encoder)-idx-1
            if idx==0:
                input=feature[-1]
            else:
                input=torch.cat([feature[-1],feature[skip_layer]],dim=1)
            out=layer(input)
            feature.append(out)
        segment_map=self.segment_layer(feature[-1])
        input=torch.cat([feature[-1],feature[0]],dim=1)
        out=self.last_decoder(input)
        return out,segment_map

class Patch_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,strides=2,firist_layer=False,last_layer=False):
        super(Patch_Conv,self).__init__()
        if firist_layer:
            self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=strides,padding=1),nn.LeakyReLU(0.2,True))
        elif last_layer:
            self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=strides,padding=1),nn.Sigmoid())
        else:
            self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=strides,padding=1),nn.BatchNorm2d(out_channels),nn.LeakyReLU(0.2,True))
    def forward(self,x):
        out=self.conv(x)
        return out
class PatchGan(nn.Module):
    def  __init__(self,input_channel):
        super(PatchGan,self).__init__()
        self.layer1=Patch_Conv(input_channel,64,firist_layer=True)
        self.layer2=Patch_Conv(64,128)
        self.layer3=Patch_Conv(128,256)
        self.layer4=Patch_Conv(256,512,1)
        self.layer5=Patch_Conv(512,1,1,last_layer=True)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        return out

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks =nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class DobuleConv(nn.Module):
    def __init__(self,incannel,out_channel):
        super(DobuleConv,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(incannel,out_channel,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),nn.BatchNorm2d(out_channel))
        self.conv2=nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),nn.BatchNorm2d(out_channel))
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        return out
class Unet2(nn.Module):
    def __init__(self):
        super(Unet2,self).__init__()
        self.conv1=DobuleConv(3,64)
        self.pool1=nn.MaxPool2d(2)
        self.conv2=DobuleConv(64,128)
        self.pool2=nn.MaxPool2d(2)
        self.conv3=DobuleConv(128,256)
        self.pool3=nn.MaxPool2d(2)
        self.conv4=DobuleConv(256,512)
        self.pool4=nn.MaxPool2d(2)
        self.conv5=DobuleConv(512,1024)
        self.up1=nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv6=DobuleConv(1024,512)
        self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv7=DobuleConv(512,256)
        self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv8=DobuleConv(256,128)
        self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv9=DobuleConv(128,64)
        self.conv10=nn.Conv2d(64,3,3,1,1)
    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up1=self.up1(c5)
        cat1=torch.cat([up1,c4],dim=1)
        c6=self.conv6(cat1)
        up2=self.up2(c6)
        cat2=torch.cat([up2,c3],dim=1)
        c7=self.conv7(cat2)
        up3=self.up3(c7)
        cat3=torch.cat([up3,c2],dim=1)
        c8=self.conv8(cat3)
        up4=self.up4(c8)
        cat4=torch.cat([up4,c1],dim=1)
        c9=self.conv9(cat4)
        c10=self.conv10(c9)
        return c10

def conv3x3(self,in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(ResBlock,self).__init__()
        self.conv1=conv3x3(in_channels,out_channels,stride)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(out_channels,out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample
    def froward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        if self.downsample:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=10):
        super(ResNet,self).__init__()
        self.in_channels=16
        self.conv=conv3x3(1,16)
        self.bn=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.layer1=self.make_layer(block,16,layers[0])
        self.layer2=self.make_layer(block,32,layers[0],2)
        self.layer3=self.make_layer(block,64,layers[1],2)
        self.avg_pool=nn.AvgPool2d(8,ceil_mode=False)
        self.fc=nn.Linear(64,10)
    def make_layer(self,block,out_chinnels,blocks,stride=1):
        downsample=None
        if (stride!=1) or (self.in_channels!=out_chinnels):
            downsample=nn.Sequential(conv3x3(self.in_channels,out_chinnels,stride=stride),nn.BatchNorm2d(out_chinnels))
        layers=[]
        layers.append(block(self.in_channels,out_chinnels,stride,downsample))
        self.in_channels=out_chinnels
        for i in range(1,blocks):
            layers.append(out_chinnels,out_chinnels)
            return nn.Sequential(*layers)
    def forwad(self,x):
        out=self.conv(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.avg_pool(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out
if __name__ =="__main__":
    from torchsummary import summary
    net=Pix2Pix(4,3).cuda()
    summary(net,(4,256,256),batch_size=16)
    # net=Unet2().cuda()
    # summary(net,(3,256,256),batch_size=1)
    # net2=PatchGan(6).cuda()
    # summary(net2,(6,256,256),batch_size=1)
