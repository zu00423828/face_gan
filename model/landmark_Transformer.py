from numpy import spacing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import ModuleList, Sequential
from torch.nn.modules.conv import ConvTranspose2d

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        encoder_layer=[]
        layer=nn.Sequential(nn.Conv2d(3,64,3,stride=2,padding=1),nn.ReLU(inplace=True),BatchNorm2d(64))
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
                layer=nn.Sequential(nn.Conv2d(64,out_channels,3,stride=2,padding=1),nn.ReLU(inplace=True),BatchNorm2d(out_channels))
            else:
                layer=nn.Sequential(nn.Conv2d(layer_specs[idx-1],out_channels,3,stride=2,padding=1),nn.ReLU(inplace=True),BatchNorm2d(out_channels))
            encoder_layer.append(layer)
        self.encoder=ModuleList(encoder_layer)
        layer_specs = [
        64 * 8,   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        64 * 8,   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        64 * 8,   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        64 * 8,   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        64 * 4,   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        64 * 2,   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        64,       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
        decoder_layer=[]
        for idx,out_channel in enumerate(layer_specs):
            layer=nn.Sequential(nn.ConvTranspose2d(out_channels,out_channels,3,stride=2,padding=1),nn.ReLU(inplace=True),BatchNorm2d(out_channels))
            decoder_layer.append(layer)
        self.decoder=ModuleList(decoder_layer)

    def forward(self,x):
        feature=[]
        out=self.encoder[0](x)
        feature.append(out)
        for layer in self.encoder[1:]:
            out=layer(out)
            feature.append(out)
        out=self.decoder[0](feature[-1])
        feature.append(out)
        for layer in self.decoder[1:]:
            input=torch.cat([feature[-1],feature[-2]],dim=1)
            out=layer(input)
        return feature[-1]
if __name__ =="__main__":
    from torchsummary import summary
    net=Unet().cuda()
    summary(net,(3,256,256))
