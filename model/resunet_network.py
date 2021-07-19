import torch.nn as nn
import torch
from torchvision.models import resnet


class PreActivateDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class PreActivateResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResUpBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.up_sample = nn.Upsample(scale_factor=2)
        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)+self.ch_avg(x)


class PreActivateResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1), 
            nn.BatchNorm2d(out_channels)
            )
        self.double_conv=PreActivateDoubleConv(in_channels,out_channels)
        self.down_sample=nn.MaxPool2d(2)
        # self.relu=nn.ReLU()

    def forward(self,x):
        identity=self.ch_avg(x)
        out=self.double_conv(x)
        out=out+identity
        return self.down_sample(out),out


class DeepResUnet(nn.Module):
    def __init__(self,out_clasess=1):
        super().__init__()
        self.down_conv1=PreActivateResBlock(1,64)
        self.down_conv2=PreActivateResBlock(64,128)
        self.down_conv3=PreActivateResBlock(128,256)
        self.down_conv4=PreActivateResBlock(256,512)
        self.double_conv=PreActivateDoubleConv(512,1024)
        self.up_conv4=PreActivateResUpBlock(512+1024,512)
        self.up_conv3=PreActivateResUpBlock(256+512,256)
        self.up_conv2=PreActivateResUpBlock(128+256,128)
        self.up_conv1=PreActivateResUpBlock(128+64,64)
        self.conv_last=nn.Conv2d(64,out_clasess,1,1,1)
    def forward(self, x):
        x,skip1_out=self.down_conv1(x)
        x,skip2_out=self.down_conv2(x)
        x,skip3_out=self.down_conv3(x)
        x,skip4_out=self.down_conv4(x)
        x=self.double_conv(x)
        x=self.up_conv4(x,skip4_out)
        x=self.up_conv3(x,skip3_out)
        x=self.up_conv2(x,skip2_out)
        x=self.up_conv1(x,skip1_out)
        x=self.conv_last(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.double_conv(x)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down_sample(out), out



class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class ResUnet(nn.Module):
    def __init__(self,in_channels,out_clasess=1):
        super(ResUnet,self).__init__()
        self.down_conv1=ResBlock(in_channels,64)
        self.down_conv2=ResBlock(64,128)
        self.down_conv3=ResBlock(128,256)
        self.down_conv4=ResBlock(256,512)
        self.double_conv=DoubleConv(512,1024)
        self.up_conv4=UpBlock(512+1024,512)
        self.up_conv3=UpBlock(256+512,256)
        self.up_conv2=UpBlock(128+256,128)
        self.up_conv1=UpBlock(128+64,64)
        self.map_layer=nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128,1,1),nn.Sigmoid())
        self.conv_last=nn.Sequential(nn.Conv2d(64,out_clasess,1),nn.Sigmoid())
    def forward(self,x):

        x,skip1_out=self.down_conv1(x)
        x,skip2_out=self.down_conv2(x)
        x,skip3_out=self.down_conv3(x)
        x,skip4_out=self.down_conv4(x)
        x=self.double_conv(x)
        x=self.up_conv4(x,skip4_out)
        x=self.up_conv3(x,skip3_out)
        x=self.up_conv2(x,skip2_out)
        map=self.map_layer(x)
        x=self.up_conv1(x,skip1_out)
        x=self.conv_last(x)
        return x,map
if __name__=="__main__":
    from torchsummary import summary

    # deepresunet=DeepResUnet().cuda()
    # summary(deepresunet,(1,256,256))

    resunet=ResUnet(4,3).cuda()
    summary(resunet,(4,256,256),batch_size=1)


