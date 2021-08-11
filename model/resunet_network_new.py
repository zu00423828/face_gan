from torch.autograd.grad_mode import F
import torch.nn as nn
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


class PreActivateDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class PreActivateResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResUpBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
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
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.ch_avg(x)
        out = self.double_conv(x)
        out = out+identity
        return self.down_sample(out), out


class DeepResUnet(nn.Module):
    def __init__(self, in_channels, out_clasess=1):
        super().__init__()
        self.down_conv1 = PreActivateResBlock(in_channels, 64)
        self.down_conv2 = PreActivateResBlock(64, 128)
        self.down_conv3 = PreActivateResBlock(128, 256)
        self.down_conv4 = PreActivateResBlock(256, 512)
        self.double_conv = PreActivateDoubleConv(512, 1024)
        self.up_conv4 = PreActivateResUpBlock(512+1024, 512)
        self.up_conv3 = PreActivateResUpBlock(256+512, 256)
        self.up_conv2 = PreActivateResUpBlock(128+256, 128)
        self.up_conv1 = PreActivateResUpBlock(128+64, 64)
        self.map_layer = nn.Sequential(nn.Upsample(
            scale_factor=2), nn.Conv2d(128, 1, 1), nn.Sigmoid())
        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_clasess, 1), nn.Tanh())
        # self.conv_last=nn.Conv2d(64,out_clasess,1,1,1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        map = self.map_layer(x)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x, map

class FetureNet(nn.Module):
    def __init__(self):
        super(FetureNet, self).__init__()
        feature = [64, 128, 256, 512,512,512,512,512]
        layers = []
        # self.feature=nn.Sequential(nn.Conv2d(in_channels=24,out_channels=16,kernel_size=3,stride=2,padding=1),)
        for i, item in enumerate(feature):
            if i == 0:
                layers.append(nn.Conv2d(24, item, 3, 2, 1))
            else:
                layers.append(nn.Conv2d(feature[i-1], item, 3, 2, 1))
            layers.append(nn.BatchNorm2d(item))
            layers.append(nn.ReLU(inplace=True))
        # self.layer=nn.ModuleList(layers)
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
            # nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.double_conv = DoubleConv(in_channels, out_channels)
        # self.down_sample = nn.MaxPool2d(2)
        # self.down_sample = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3,
        #                                  stride=2, padding=1, bias=False),nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return out
        # return self.down_sample(out), out


class UpBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UpBlock, self).__init__()
        # self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_sample = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       # nn.InstanceNorm2d(out_channels),
                                       nn.ReLU(inplace=True))
        # self.up_sample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(
        #     in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        #self.identity=nn.Sequential(nn.Conv2d(mid_channels,out_channels,1,1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.double_conv = DoubleConv(mid_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)#+self.identity(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        # self.down_sample = nn.MaxPool2d(2)
        self.down_sample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                         stride=2, padding=1, bias=False),
                                         nn.BatchNorm2d(out_channels),
                                         #  nn.InstanceNorm2d(out_channels),
                                         nn.ReLU(True))

    def forward(self, input):
        return self.down_sample(input)


class ResUnet(nn.Module):
    def __init__(self, in_channels, out_clasess=1):
        super(ResUnet, self).__init__()
        encode_block = []
        down_block = []
        # block = [64, 128, 256, 512, 512, 512, 512, 512]
        block = [32,64, 128, 256, 512]
        # block = [16,32,64,128,256,512,1024,1024]
        for i in range(len(block)):
            if i == 0:
                encode_block.append(ResBlock(in_channels, block[i]))
            else:
                encode_block.append(ResBlock(block[i-1], block[i]))
            down_block.append(DownBlock(block[i], block[i]))
        self.encode_conv_block = nn.ModuleList(encode_block)
        self.down_block = nn.ModuleList(down_block)
        # self.other_feateue=FetureNet()
        self.double_conv = DoubleConv(block[-1], block[-1]*2)

        # block = [(1024, 1024, 512), 
        #             (512, 1024, 512), (512, 1024,512), (512, 1024, 512),(512, 1024, 512), (512, 512, 256), (256, 256, 128), (128, 128, 64)]
        # block = [(2048, 2048, 1024), 
        #             (1024,2048,1024), (1024,1024,512), (512,512,256),(256,256,128), (128,128,64), (64, 64,32), (32, 32, 16)]
        block = [(1024, 1024, 512), 
                    (512,512,256), (256,256,128), (128,128,64),(64,64,32)]
        up_block=[]
        for item in block:
            up_block.append(UpBlock(item[0],item[1],item[2]))
        self.up_block=nn.ModuleList(up_block)




        self.map_layer = nn.Sequential(
            # nn.Upsample(scale_factor=2), nn.Conv2d(128, 1, 1),
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid())
        self.conv_last = nn.Sequential(
            nn.Conv2d(32+1, out_clasess, 1), nn.Tanh())

    def forward(self, x):
        encode = []
        for i in range(len(self.encode_conv_block)):
            encode.append(self.encode_conv_block[i](x))
            x = self.down_block[i](encode[-1])
        x = self.double_conv(x)
        map=None
        upblock_len=len(self.up_block)
        for i in range(upblock_len):   
            x=self.up_block[i](x,encode[-1])
            encode.pop()
            if i==upblock_len-2:
                map=self.map_layer(x)
        last_skip=torch.cat([x,map],dim=1)
        x=self.conv_last(last_skip)
        return x,map




def weight_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight, 0.5, 0.166)
        # m.weight.data.fill_(0.5)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 0.5, 0.166)
        torch.nn.init.zeros_(m.bias)


if __name__ == "__main__":
    from torchsummary import summary

    # deepresunet=DeepResUnet().cuda()
    # summary(deepresunet,(1,256,256))

    resunet = ResUnet(6, 3)
    # net = FetureNet()
    # summary(net,(24,256,256),batch_size=1,device="cpu")
    # print(net)
    # summary(net,(36,256,256),device="cpu")
    # resunet.apply(weight_init)
    # # deepresunet=DeepResUnet(4,3)
    # # summary(deepresunet,(4,256,256),batch_size=1,device="cpu")
    # summary(resunet, [(6, 256, 256),(24,256,256)], batch_size=1, device="cpu")
    summary(resunet, (6, 256, 256), batch_size=1, device="cpu")