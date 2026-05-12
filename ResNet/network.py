import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.nn import Sequential


#layer 50

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, stride=1, kernel_size=7, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        #Convolution 2
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.conv2_1 = nn.Conv2d(64,64, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64,256,kernel_size=1, stride=1, padding='same', bias=False)
        self.bn2_3 = nn.BatchNorm2d(256)

        self.conv2_4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn2_4 = nn.BatchNorm2d(64)
        self.conv2_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2_5 = nn.BatchNorm2d(64)
        self.conv2_6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn2_6 = nn.BatchNorm2d(256)

        self.out2_conv = nn.Conv2d(256,256,kernel_size=1, stride=2, bias=False)
        self.conv2_7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn2_7 = nn.BatchNorm2d(64)
        self.conv2_8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2_8 = nn.BatchNorm2d(64)
        self.conv2_9 = nn.Conv2d(64, 256, kernel_size=1, stride=2, bias=False)
        self.bn2_9 = nn.BatchNorm2d(256)

        #Convolution 3
        self.conv3_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn3_3 = nn.BatchNorm2d(512)

        self.conv3_4 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn3_4 = nn.BatchNorm2d(128)
        self.conv3_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3_5 = nn.BatchNorm2d(128)
        self.conv3_6 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn3_6 = nn.BatchNorm2d(512)

        self.conv3_7 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn3_7 = nn.BatchNorm2d(128)
        self.conv3_8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3_8 = nn.BatchNorm2d(128)
        self.conv3_9 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn3_9 = nn.BatchNorm2d(512)

        self.out5_conv = nn.Conv2d(512, 512, kernel_size=1, stride=2, bias=False)
        self.conv3_10 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn3_10 = nn.BatchNorm2d(128)
        self.conv3_11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3_11 = nn.BatchNorm2d(128)
        self.conv3_12 = nn.Conv2d(128, 512, kernel_size=1, stride=2, bias=False)
        self.bn3_12 = nn.BatchNorm2d(512)

        #Convolution 4
        self.conv4_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.conv4_3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_3 = nn.BatchNorm2d(1024)

        self.conv4_4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_4 = nn.BatchNorm2d(256)
        self.conv4_5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4_5 = nn.BatchNorm2d(256)
        self.conv4_6 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_6 = nn.BatchNorm2d(1024)

        self.conv4_7 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_7 = nn.BatchNorm2d(256)
        self.conv4_8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4_8 = nn.BatchNorm2d(256)
        self.conv4_9 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_9 = nn.BatchNorm2d(1024)

        self.conv4_10 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_10 = nn.BatchNorm2d(256)
        self.conv4_11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4_11 = nn.BatchNorm2d(256)
        self.conv4_12 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_12 = nn.BatchNorm2d(1024)

        self.conv4_13 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_13 = nn.BatchNorm2d(256)
        self.conv4_14 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4_14 = nn.BatchNorm2d(256)
        self.conv4_15 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_15 = nn.BatchNorm2d(1024)

        self.out10_conv = nn.Conv2d(1024, 1024, kernel_size=1, stride=2, bias=False)
        self.conv4_16 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn4_16 = nn.BatchNorm2d(256)
        self.conv4_17 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4_17 = nn.BatchNorm2d(256)
        self.conv4_18 = nn.Conv2d(256, 1024, kernel_size=1, stride=2, bias=False)
        self.bn4_18 = nn.BatchNorm2d(1024)

        #Convolution 5
        self.conv5_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn5_3 = nn.BatchNorm2d(2048)

        self.conv5_4 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.conv5_5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn5_5 = nn.BatchNorm2d(512)
        self.conv5_6 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn5_6 = nn.BatchNorm2d(2048)

        self.out12_conv = nn.Conv2d(2048, 2048, kernel_size=1, stride=2, bias=False)
        self.conv5_7 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn5_7 = nn.BatchNorm2d(512)
        self.conv5_8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn5_8 = nn.BatchNorm2d(512)
        self.conv5_9 = nn.Conv2d(512, 2048, kernel_size=1, stride=2, bias=False)
        self.bn5_9 = nn.BatchNorm2d(2048)

        self.avgpool1 = nn.AvgPool2d(4, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 200)

    def forward(self,x):
        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        #print(out.shape)

        #conv2
        out = F.relu((self.bn2_1(self.conv2_1(out))))
        out = F.relu((self.bn2_2(self.conv2_2(out))))
        out = F.relu((self.bn2_3(self.conv2_3(out))))
        #print(out.shape)
        out1 = out
        out = F.relu((self.bn2_4(self.conv2_4(out))))
        out = F.relu((self.bn2_5(self.conv2_5(out))))
        out = self.bn2_6(self.conv2_6(out))
        out = F.relu(out + out1)
        #print(out.shape)
        out2 = self.out2_conv(out)
        out = F.relu((self.bn2_7(self.conv2_7(out))))
        out = F.relu((self.bn2_8(self.conv2_8(out))))
        out = self.bn2_9(self.conv2_9(out))
        out = F.relu(out + out2)
        #print(out.shape)

        #conv3
        out = F.relu((self.bn3_1(self.conv3_1(out))))
        out = F.relu((self.bn3_2(self.conv3_2(out))))
        out = F.relu((self.bn3_3(self.conv3_3(out))))
        #print(out.shape)
        out3 = out
        out = F.relu((self.bn3_4(self.conv3_4(out))))
        out = F.relu((self.bn3_5(self.conv3_5(out))))
        out = self.bn3_6(self.conv3_6(out))
        out = F.relu(out + out3)
        #print(out.shape)
        out4 = out
        out = F.relu((self.bn3_7(self.conv3_7(out))))
        out = F.relu((self.bn3_8(self.conv3_8(out))))
        out = self.bn3_9(self.conv3_9(out))
        out = F.relu(out + out4)
        #print(out.shape)
        out5 = self.out5_conv(out)
        out = F.relu((self.bn3_10(self.conv3_10(out))))
        out = F.relu((self.bn3_11(self.conv3_11(out))))
        out = self.bn3_12(self.conv3_12(out))
        out = F.relu(out + out5)
        #print(out.shape)

        #conv4
        out = F.relu((self.bn4_1(self.conv4_1(out))))
        out = F.relu((self.bn4_2(self.conv4_2(out))))
        out = F.relu((self.bn4_3(self.conv4_3(out))))
        #print(out.shape)
        out6 = out
        out = F.relu((self.bn4_4(self.conv4_4(out))))
        out = F.relu((self.bn4_5(self.conv4_5(out))))
        out = self.bn4_6(self.conv4_6(out))
        out = F.relu(out + out6)
        #print(out.shape)
        out7 = out
        out = F.relu((self.bn4_7(self.conv4_7(out))))
        out = F.relu((self.bn4_8(self.conv4_8(out))))
        out = self.bn4_9(self.conv4_9(out))
        out = F.relu(out + out7)
        #print(out.shape)
        out8 = out
        out = F.relu((self.bn4_10(self.conv4_10(out))))
        out = F.relu((self.bn4_11(self.conv4_11(out))))
        out = self.bn4_12(self.conv4_12(out))
        out = F.relu(out + out8)
        #print(out.shape)
        out9 = out
        out = F.relu((self.bn4_13(self.conv4_13(out))))
        out = F.relu((self.bn4_14(self.conv4_14(out))))
        out = self.bn4_15(self.conv4_15(out))
        out = F.relu(out + out9)
        #print(out.shape)
        out10 = self.out10_conv(out)
        out = F.relu((self.bn4_16(self.conv4_16(out))))
        out = F.relu((self.bn4_17(self.conv4_17(out))))
        out = self.bn4_18(self.conv4_18(out))
        out = F.relu(out + out10)
        #print(out.shape)

        #conv5
        out = F.relu((self.bn5_1(self.conv5_1(out))))
        out = F.relu((self.bn5_2(self.conv5_2(out))))
        out = F.relu((self.bn5_3(self.conv5_3(out))))
        #print(out.shape)
        out11 = out
        out = F.relu((self.bn5_4(self.conv5_4(out))))
        out = F.relu((self.bn5_5(self.conv5_5(out))))
        out = self.bn5_6(self.conv5_6(out))
        out = F.relu(out + out11)
        #print(out.shape)
        out12 = self.out12_conv(out)
        out = F.relu((self.bn5_7(self.conv5_7(out))))
        out = F.relu((self.bn5_8(self.conv5_8(out))))
        out = self.bn5_9(self.conv5_9(out))
        out = F.relu(out + out12)
        #print(out.shape)

        out = self.avgpool1(out)
        # print(out.shape)

        out = self.flatten(out)
#         print(out.shape)
        out = self.fc1(out)
#         print(out.shape)

        return out


#layer 101
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
        self.downsample0 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channel)


    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.in_channel == self.out_channel and self.stride == 1:
            out = F.relu(out + x)
        elif self.in_channel != self.out_channel and self.stride != 1:
            x = self.bn4(self.downsample0(x))
            out = F.relu(out + x)
        else:
            out = F.relu(out)
        return out

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
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
        self.block3_7 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_8 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_9 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_10 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_11 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_12 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_13 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_14 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_15 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_16 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_17 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_18 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_19 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_20 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_21 = ResBlock(1024, 1024, 256, 1, 'same')
        self.block3_22 = ResBlock(1024, 1024, 256, 1, 'same')
        # #self.downsample2 = nn.Conv2d(1024,1024, kernel_size=1, stride=2, bias=False)
        self.block3_23 = ResBlock(1024, 1024, 256, 1, 'same') #[64,1024,8,8]

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
        out = self.block3_7(out)
        # #print(out.shape)
        out = self.block3_8(out)
        # #print(out.shape)
        out = self.block3_9(out)
        # #print(out.shape)
        out = self.block3_10(out)
        # #print(out.shape)
        out = self.block3_11(out)
        # #print(out.shape)
        out = self.block3_12(out)
        # #print(out.shape)
        out = self.block3_13(out)
        #print(out.shape)
        out = self.block3_14(out)
        # #print(out.shape)
        out = self.block3_15(out)
        # #print(out.shape)
        out = self.block3_16(out)
        # #print(out.shape)
        out = self.block3_17(out)
        # #print(out.shape)
        out = self.block3_18(out)
        # #print(out.shape)
        out = self.block3_19(out)
        # #print(out.shape)
        out = self.block3_20(out)
        # #print(out.shape)
        out = self.block3_21(out)
        # #print(out.shape)
        out = self.block3_22(out)
        # #print(out.shape)
        out = self.block3_23(out)
        #print('end Conv4 {}'.format(out.shape))

        #Conv5
        out = self.block4_1(out)
        #print(out.shape)
        out = self.block4_2(out)
        #print(out.shape)
        out = self.block4_3(out)
        #print('end Conv5 {}'.format(out.shape))

        out = self.avgpool1(out)
        out = self.flatten(out)
        #print('end flatten {}'.format(out.shape))

        out = self.dropout(out)
        out = self.fc1(out)
        return out
