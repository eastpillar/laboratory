import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch

class FCN(nn.Module):
    def __init__(self,VGG_net):
        super(FCN,self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(2, stride=2)
        # self.conv1_1 = VGG_net[0]
        # self.bn1 = VGG_net[1]
        # self.conv1_2 = VGG_net[3]
        # self.bn2 = VGG_net[4]
        # self.maxpool1 = VGG_net[6]  # 128
        # self.conv2_1 = VGG_net[7]
        # self.bn3 = VGG_net[8]
        # self.conv2_2 = VGG_net[10]
        # self.bn4 = VGG_net[11]
        # self.maxpool2 = VGG_net[13]  # 64
        # self.conv3_1 = VGG_net[14]
        # self.bn5 = VGG_net[15]
        # self.conv3_2 = VGG_net[17]
        # self.bn6 = VGG_net[18]
        # self.conv3_3 = VGG_net[20]
        # self.bn7 = VGG_net[21]
        # self.maxpool3 = VGG_net[23]  # 32
        # self.conv4_1 = VGG_net[24]
        # self.bn8 = VGG_net[25]
        # self.conv4_2 = VGG_net[27]
        # self.bn9 = VGG_net[28]
        # self.conv4_3 = VGG_net[30]
        # self.bn10 = VGG_net[31]
        # self.maxpool4 = VGG_net[33]  # 16
        # self.conv5_1 = VGG_net[34]
        # self.bn11 = VGG_net[35]
        # self.conv5_2 = VGG_net[37]
        # self.bn12 = VGG_net[38]
        # self.conv5_3 = VGG_net[40]
        # self.bn13 = VGG_net[41]
        # self.maxpool5 = VGG_net[43]  # 8

        self.conv_fc1 = nn.Conv2d(512, 4096, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_fc1 = nn.BatchNorm2d(4096)
        # self.drop1 = nn.Dropout(p=0.5)
        self.conv_fc2 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_fc2 = nn.BatchNorm2d(4096)
        self.drop2 = nn.Dropout(p=0.5)
        self.conv_fc3 = nn.Conv2d(4096, 21, kernel_size=1, stride=1, padding=0, bias=False)

        # self.resize_FCNs = nn.Upsample((256,256), mode='bilinear')
        # self.resize_2s = nn.Upsample((16,16), mode='bilinear')
        # self.resize_2s_2 = nn.Upsample((32,32), mode='bilinear')
        #self.upsampleX32 = nn.ConvTranspose2d(22,22,kernel_size=32, stride=32, bias = False)
        #self.upsampleX16 = nn.ConvTranspose2d(22,22,kernel_size=16, stride=16, bias = False)
        self.pool3_conv = nn.Conv2d(256, 21, kernel_size=1, stride=1, bias=False)
        self.pool4_conv = nn.Conv2d(512,21,kernel_size=1, stride=1, bias=False)

        self.upsampleX8 = nn.ConvTranspose2d(21,21, kernel_size=16, stride=8, padding=4, bias = False)

        self.upsampleX2_1 = nn.ConvTranspose2d(21,21, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampleX2_2 = nn.ConvTranspose2d(21,21, kernel_size=4, stride=2, padding=1, bias=False)


    def forward(self,x):
        #out = self.features(x)
        out = F.relu(self.bn1(self.conv1_1(x)))
        out = F.relu(self.bn2(self.conv1_2(out)))
        out = self.maxpool1(out)

        out = F.relu(self.bn3(self.conv2_1(out)))
        out = F.relu(self.bn4(self.conv2_2(out)))
        out = self.maxpool2(out)

        out = F.relu(self.bn5(self.conv3_1(out)))
        out = F.relu(self.bn6(self.conv3_2(out)))
        out = F.relu(self.bn7(self.conv3_3(out)))
        out = self.maxpool3(out)
        pool3_pred = self.pool3_conv(out)

        out = F.relu(self.bn8(self.conv4_1(out)))
        out = F.relu(self.bn9(self.conv4_2(out)))
        out = F.relu(self.bn10(self.conv4_3(out)))
        out = self.maxpool4(out)
        pool4_pred = self.pool4_conv(out)

        out = F.relu(self.bn11(self.conv5_1(out)))
        out = F.relu(self.bn12(self.conv5_2(out)))
        out = F.relu(self.bn13(self.conv5_3(out)))
        out = self.maxpool5(out)

        out = F.relu(self.bn_fc1(self.conv_fc1(out)))
        # out = self.drop1(out)
        out = F.relu(self.bn_fc2(self.conv_fc2(out)))
        out = self.drop2(out)

        pool5_out = self.conv_fc3(out) #[4096 -> 22 channel]
        #FCN_32s = self.upsampleX32(pool5_out)
        upsample1_1 = self.upsampleX2_1(pool5_out)
        out = pool4_pred + upsample1_1

        #FCN_16s = self.upsampleX16(out)
        upsample1_2 = self.upsampleX2_2(out)

        out = pool3_pred + upsample1_2
        FCN_8s = self.upsampleX8(out)


        return FCN_8s