import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.nn import Sequential
import torch


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride, padding):
        super(BottleNeck,self).__init__()
        mid_channel = out_channel * 4 #out_channel == growth rate
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=False)

        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))

        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channel, out_channel): #64, 32
        super(DenseBlock, self).__init__()
        mid_channel = out_channel * 4 #mid = 128, out = 32

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel,out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        #self.downsample0 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)


    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        #print("before concat {}".format(out.shape))
        out = torch.cat([out, x], 1)
        #print("in Denseblock {}".format(out.shape))

        return out


class TranBlock(nn.Module):
    def __init__(self, in_channel, comp_factor):
        super(TranBlock,self).__init__()
        out_channel = int(comp_factor * in_channel)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.avgpool(out)

        return out



class DenseNet121(nn.Module):
    def __init__(self, growth_rate, comp_factor):
        super(DenseNet121, self).__init__()
        self.growth_rate = growth_rate
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 2*growth_rate, stride=2, kernel_size=7, padding=3, bias=False) #64
        self.maxpool1 = nn.MaxPool2d(3, 2, padding=1) #32

        #DenseBlock 1
        self.block1_1 = DenseBlock(2*growth_rate,32)
        self.block1_2 = DenseBlock(96,32)
        self.block1_3 = DenseBlock(128,32)
        self.block1_4 = DenseBlock(160, 32)
        self.block1_5 = DenseBlock(192, 32)
        self.block1_6 = DenseBlock(224, 32)
        self.trans1 = TranBlock(256, comp_factor) #16

        #DenseBlock 2
        self.block2_1 = DenseBlock(128,32)
        self.block2_2 = DenseBlock(160, 32)
        self.block2_3 = DenseBlock(192, 32)
        self.block2_4 = DenseBlock(224, 32)
        self.block2_5 = DenseBlock(256, 32)
        self.block2_6 = DenseBlock(288, 32)
        self.block2_7 = DenseBlock(320, 32)
        self.block2_8 = DenseBlock(352, 32)
        self.block2_9 = DenseBlock(384, 32)
        self.block2_10 = DenseBlock(416, 32)
        self.block2_11 = DenseBlock(448, 32)
        self.block2_12 = DenseBlock(480, 32)
        self.trans2 = TranBlock(512, comp_factor) #8
        #convolution 4
        self.block3_1 = DenseBlock(256, 32)
        self.block3_2 = DenseBlock(288, 32)
        self.block3_3 = DenseBlock(320, 32)
        self.block3_4 = DenseBlock(352, 32)
        self.block3_5 = DenseBlock(384, 32)
        self.block3_6 = DenseBlock(416, 32)
        self.block3_7 = DenseBlock(448, 32)
        self.block3_8 = DenseBlock(480, 32)
        self.block3_9 = DenseBlock(512, 32)
        self.block3_10 = DenseBlock(544, 32)
        self.block3_11 = DenseBlock(576, 32)
        self.block3_12 = DenseBlock(608, 32)
        self.block3_13 = DenseBlock(640, 32)
        self.block3_14 = DenseBlock(672, 32)
        self.block3_15 = DenseBlock(704, 32)
        self.block3_16 = DenseBlock(736, 32)
        self.block3_17 = DenseBlock(768, 32)
        self.block3_18 = DenseBlock(800, 32)
        self.block3_19 = DenseBlock(832, 32)
        self.block3_20 = DenseBlock(864, 32)
        self.block3_21 = DenseBlock(896, 32)
        self.block3_22 = DenseBlock(928, 32)
        self.block3_23 = DenseBlock(960, 32) #[64,32,8,8]
        self.block3_24 = DenseBlock(992, 32)
        self.trans3 = TranBlock(1024, comp_factor) #4

        #convolution 5
        self.block4_1 = DenseBlock(512, 32)
        self.block4_2 = DenseBlock(544, 32)
        self.block4_3 = DenseBlock(576, 32)
        self.block4_4 = DenseBlock(608, 32)
        self.block4_5 = DenseBlock(640, 32)
        self.block4_6 = DenseBlock(672, 32)
        self.block4_7 = DenseBlock(704, 32)
        self.block4_8 = DenseBlock(736, 32)
        self.block4_9 = DenseBlock(768, 32)
        self.block4_10 = DenseBlock(800, 32)
        self.block4_11 = DenseBlock(832, 32)
        self.block4_12 = DenseBlock(864, 32)
        self.block4_13 = DenseBlock(896, 32)
        self.block4_14 = DenseBlock(928, 32)
        self.block4_15 = DenseBlock(960, 32)
        self.block4_16 = DenseBlock(992, 32)

        self.avgpool1 = nn.AvgPool2d(4, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024, 200)

    def forward(self,x):
        #print('growth rate {}'.format(self.growth_rate))
        # print(x.shape)
        out = self.conv1(F.relu(self.bn1(x)))
#         print(out.shape)
        out = self.maxpool1(out)
#         print('end pooling {}'.format(out.shape))

        #DenseBlock 1
        out = self.block1_1(out)
#         print(out.shape)
        out = self.block1_2(out)
#         print(out.shape)
        out = self.block1_3(out)
#         print(out.shape)
        out = self.block1_4(out)
#         print(out.shape)
        out = self.block1_5(out)
#         print(out.shape)
        out = self.block1_6(out)
#         print(out.shape)
        out = self.trans1(out)
#         print('end DenseBlock 1 {}'.format(out.shape))

        #DenseBlock 2
        out = self.block2_1(out)
#         print(out.shape)
        out = self.block2_2(out)
#         print(out.shape)
        out = self.block2_3(out)
#         print(out.shape)
        out = self.block2_4(out)
#         print(out.shape)
        out = self.block2_5(out)
#         print(out.shape)
        out = self.block2_6(out)
#         print(out.shape)
        out = self.block2_7(out)
#         print(out.shape)
        out = self.block2_8(out)
#         print(out.shape)
        out = self.block2_9(out)
#         print(out.shape)
        out = self.block2_10(out)
#         print(out.shape)
        out = self.block2_11(out)
#         print(out.shape)
        out = self.block2_12(out)
#         print(out.shape)
        out = self.trans2(out)
#         print('end DenseBlock 2 {}'.format(out.shape))

        #DenseBlock 3
        out = self.block3_1(out)
#         print(out.shape)
        out = self.block3_2(out)
#         print(out.shape)
        out = self.block3_3(out)
#         print(out.shape)
        out = self.block3_4(out)
#         print(out.shape)
        out = self.block3_5(out)
#         print(out.shape)
        out = self.block3_6(out)
#         print(out.shape)
        out = self.block3_7(out)
#         print(out.shape)
        out = self.block3_8(out)
#         print(out.shape)
        out = self.block3_9(out)
#         print(out.shape)
        out = self.block3_10(out)
#         print(out.shape)
        out = self.block3_11(out)
#         print(out.shape)
        out = self.block3_12(out)
#         print(out.shape)
        out = self.block3_13(out)
#         print(out.shape)
        out = self.block3_14(out)
#         print(out.shape)
        out = self.block3_15(out)
#         print(out.shape)
        out = self.block3_16(out)
#         print(out.shape)
        out = self.block3_17(out)
#         print(out.shape)
        out = self.block3_18(out)
#         print(out.shape)
        out = self.block3_19(out)
#         print(out.shape)
        out = self.block3_20(out)
#         print(out.shape)
        out = self.block3_21(out)
#         print(out.shape)
        out = self.block3_22(out)
#         print(out.shape)
        out = self.block3_23(out)
#         print(out.shape)
        out = self.block3_24(out)
#         print(out.shape)
        out = self.trans3(out)
#         print('end DenseBlock 3 {}'.format(out.shape))

        #DenseBlock4
        out = self.block4_1(out)
#         print(out.shape)
        out = self.block4_2(out)
#         print(out.shape)
        out = self.block4_3(out)
#         print(out.shape)
        out = self.block4_4(out)
#         print(out.shape)
        out = self.block4_5(out)
#         print(out.shape)
        out = self.block4_6(out)
#         print(out.shape)
        out = self.block4_7(out)
#         print(out.shape)
        out = self.block4_8(out)
#         print(out.shape)
        out = self.block4_9(out)
#         print(out.shape)
        out = self.block4_10(out)
#         print(out.shape)
        out = self.block4_11(out)
#         print(out.shape)
        out = self.block4_12(out)
#         print(out.shape)
        out = self.block4_13(out)
#         print(out.shape)
        out = self.block4_14(out)
#         print(out.shape)
        out = self.block4_15(out)
#         print(out.shape)
        out = self.block4_16(out)
#         print('end DenseBlock 4 {}'.format(out.shape))
        out = self.avgpool1(out)
#         print('end avg Pooling {}'.format(out.shape))
        out = self.flatten(out)
#         print('end flatten {}'.format(out.shape))
        out = self.dropout(out)
        out = self.fc1(out)
#         print('end fully connected {}'.format(out.shape))

        return out
