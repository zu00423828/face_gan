import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
class Unet(nn.Module):
    def __init__(self,out_channel):
        super(Unet,self).__init__()
        self.out_channel=out_channel
        encoder_layer=[]
        layer=nn.Sequential(nn.Conv2d(3,64,3,stride=2,padding=1),nn.ReLU(inplace=True),nn.BatchNorm2d(64))
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
        self.last_decoder=nn.Sequential(nn.ConvTranspose2d(128,3,4,stride=2,padding=1),nn.Tanh())
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