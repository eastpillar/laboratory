import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
from torch.nn import Sequential


class ChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super(ChannelAttention,self).__init__()
        out_channel = in_channel // 16
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.MLP = nn.Sequential(nn.Conv2d(in_channel, in_channel // reduction_ratio, kernel_size=1),
        #                          nn.ReLU(),
        #                          nn.Conv2d(in_channel // reduction_ratio, in_channel, kernel_size=1))
        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_channel, out_channel),
                                 nn.ReLU(),
                                 nn.Linear(out_channel, in_channel))


    def forward(self,x):
        #print(x.shape) #[256, 256, 32, 32]
        bth, ch, _, _ = x.shape
        out_max = self.maxpool(x) #[256, 256, 1, 1]
        #print(out_max.shape)
        out_max = self.MLP(out_max).view([bth, ch, 1, 1]) #[256, 256, 1, 1]
        #print(out_max.shape)
        out_avg = self.avgpool(x) #[256, 256, 1, 1]
        #print(out_avg.shape)
        out_avg = self.MLP(out_avg).view([bth, ch, 1, 1]) #[256, 256, 1, 1]
        #print(out_avg.shape)
        out = F.sigmoid(out_max + out_avg) #[256, 256, 1, 1]
        #print(out.shape)
        out = out * x #[256, 256, 32, 32]
        #print(out.shape)
        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=7,stride=1,padding=3, bias=False)

    def forward(self,x):
        #print(x.shape) #[64,256,32,32]
        bth, _, w, h = x.shape
        outMax = torch.max(x, dim=1)[0].unsqueeze(1) #[256, 1, 32, 32]
        #print(outMax.shape)
        outAvg = torch.mean(x, dim=1).unsqueeze(1) #[256, 1, 32, 32]
        #print(outAvg.shape)
        out = torch.cat((outMax, outAvg), dim=1) #[256, 2, 32, 32]
        #print(out.shape)
        out = F.sigmoid(self.conv(out)) #[64, 1, 32, 32]
        out = out * x #[64, 256, 32, 32]

        return out

class CBAM(nn.Module):
    def __init__(self,in_channel):
        super(CBAM,self).__init__()
        self.ch_atten = ChannelAttention(in_channel)
        self.sp_stten = SpatialAttention()

    def forward(self, x):
        out = self.ch_atten(x)
        out = self.sp_stten(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, stride, padding):
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel,mid_channel, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel,out_channel, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.cbam = CBAM(out_channel)
        self.downsample0 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.cbam(out)
        if self.in_channel == self.out_channel and self.stride == 1:
            out = F.relu(out + x)
        elif self.in_channel != self.out_channel and self.stride != 1:
            x = self.bn4(self.downsample0(x))
            out = F.relu(out + x)
        return out

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, stride=2, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(3, 2, padding=1)

        #convolution 2
        self.block1_1 = ResBlock(64,256,64,1,'same')
        self.block1_2 = ResBlock(256,256,64,1, 'same')
        #self.downsample0 = nn.Conv2d(256,256,kernel_size=1, stride=2, bias=False)
        self.block1_3 = ResBlock(256,256,64,1,'same') #[64,256,32,32]

        #convolution 3
        self.block2_1 = ResBlock(256,512,128,2,1)
        self.block2_2 = ResBlock(512, 512, 128, 1, 'same')
        self.block2_3 = ResBlock(512, 512, 128, 1, 'same')
        #self.downsample1 = nn.Conv2d(512,512, kernel_size=1, stride=2, bias=False)
        self.block2_4 = ResBlock(512, 512, 128, 1, 'same') #[64,512,16,16]

        #convolution 4
        self.block3_1 = ResBlock(512, 1024, 256, 2, 1)
        self.block3_2 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_3 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_4 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_5 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_6 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_7 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_8 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_9 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_10 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_11 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_12 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_13 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_14 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_15 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_16 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_17 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_18 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_19 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_20 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_21 = ResBlock(1024, 1024, 256, 1, 'same')
        # self.block3_22 = ResBlock(1024, 1024, 256, 1, 'same')
        # #self.downsample2 = nn.Conv2d(1024,1024, kernel_size=1, stride=2, bias=False)
        # self.block3_23 = ResBlock(1024, 1024, 256, 1, 'same') #[64,1024,8,8]

        #convolution 5
        self.block4_1 = ResBlock(1024, 2048, 512, 2, 1) #[64,2048,4,4]
        self.block4_2 = ResBlock(2048, 2048, 512, 1, 'same')
        #self.downsample3 = nn.Conv2d(2048, 2048, kernel_size=1, stride=2, bias=False)
        self.block4_3 = ResBlock(2048, 2048, 512, 1, 'same')

        self.avgpool1 = nn.AvgPool2d(4, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 200)

    def forward(self,x):
        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))

        #print(out.shape)
        out = self.maxpool1(out)
        #print('end pooling {}'.format(out.shape))

        #Conv2
        out = self.block1_1(out)
        #print(out.shape)
        out = self.block1_2(out)
        #print(out.shape)
        out = self.block1_3(out)
        #print('end Conv2 {}'.format(out.shape))

        #Conv3
        out = self.block2_1(out)
        #print(out.shape)
        out = self.block2_2(out)
        #print(out.shape)
        out = self.block2_3(out)
        #print(out.shape)
        out = self.block2_4(out)
        #print('end Conv3 {}'.format(out.shape))

        #Conv4
        out = self.block3_1(out)
        #print(out.shape)
        out = self.block3_2(out)
        #print(out.shape)
        out = self.block3_3(out)
        #print(out.shape)
        out = self.block3_4(out)
        #print(out.shape)
        out = self.block3_5(out)
        #print(out.shape)
        out = self.block3_6(out)
        #print(out.shape)
        # out = self.block3_7(out)
        # #print(out.shape)
        # out = self.block3_8(out)
        # #print(out.shape)
        # out = self.block3_9(out)
        # #print(out.shape)
        # out = self.block3_10(out)
        # #print(out.shape)
        # out = self.block3_11(out)
        # #print(out.shape)
        # out = self.block3_12(out)
        # #print(out.shape)
        # out = self.block3_13(out)
        # #print(out.shape)
        # out = self.block3_14(out)
        # #print(out.shape)
        # out = self.block3_15(out)
        # #print(out.shape)
        # out = self.block3_16(out)
        # #print(out.shape)
        # out = self.block3_17(out)
        # #print(out.shape)
        # out = self.block3_18(out)
        # #print(out.shape)
        # out = self.block3_19(out)
        # #print(out.shape)
        # out = self.block3_20(out)
        # #print(out.shape)
        # out = self.block3_21(out)
        # #print(out.shape)
        # out = self.block3_22(out)
        # #print(out.shape)
        # out = self.block3_23(out)
        #print('end Conv4 {}'.format(out.shape))

        #Conv5
        out = self.block4_1(out)
        #print(out.shape)
        out = self.block4_2(out)
        #print(out.shape)
        out = self.block4_3(out)
        visual_image = out

        #print('end Conv5 {}'.format(out.shape))

        out = self.avgpool1(out)
        out = self.flatten(out)
        #print('end flatten {}'.format(out.shape))

        out = self.dropout(out)
        out = self.fc1(out)
        return out, visual_image