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
