import torch
import torch.nn as nn
from torch.nn.modules import pooling
from torch.nn.modules.container import ModuleList
from torchvision import models, transforms
class Unet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Unet,self).__init__()
        self.out_channel=out_channel
        encoder_layer=[]
        layer=nn.Sequential(nn.Conv2d(in_channel,64,3,stride=2,padding=1),nn.ReLU(inplace=True),nn.BatchNorm2d(64))
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
                layer=nn.Sequential(nn.Conv2d(64,out_channels,4,stride=2,padding=1),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
            else:
                layer=nn.Sequential(nn.Conv2d(layer_specs[idx-1],out_channels,4,stride=2,padding=1),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
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
                layer=nn.Sequential(nn.ConvTranspose2d(out_channels,out_channels,4,stride=2,padding=1),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
            else:
                layer=nn.Sequential(nn.ConvTranspose2d(layer_specs[idx-1]*2,out_channels,4,stride=2,padding=1),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
            decoder_layer.append(layer)
        self.decoder=ModuleList(decoder_layer)
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
        input=torch.cat([feature[-1],feature[0]],dim=1)
        out=self.last_decoder(input)
        return out

class Patch_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,strides=2,last_layer=False):
        super(Patch_Conv,self).__init__()
        if last_layer:
            self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=strides,padding=1),nn.BatchNorm2d(out_channels),nn.Sigmoid())
        else:
            self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=strides,padding=1),nn.BatchNorm2d(out_channels),nn.ReLU(True))
    def forward(self,x):
        out=self.conv(x)
        return out
class PatchGan(nn.Module):
    def  __init__(self,input_channel):
        super(PatchGan,self).__init__()
        self.layer1=Patch_Conv(input_channel,64)
        self.layer2=Patch_Conv(64,128)
        self.layer3=Patch_Conv(128,256)
        self.layer4=Patch_Conv(256,512,1)
        self.layer5=Patch_Conv(512,1,1,True)
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






if __name__ =="__main__":
    from torchsummary import summary
    net=Unet(3,3).cuda()
    summary(net,(3,256,256),batch_size=1)
    # net=Unet2().cuda()
    # summary(net,(3,256,256),batch_size=1)
    # net2=PatchGan(6).cuda()
    # summary(net2,(6,256,256),batch_size=1)
