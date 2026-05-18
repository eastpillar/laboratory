import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self,VGG_net):
        super(UNet,self).__init__()
        self.conv1_1 = VGG_net[0]
        self.bn1 = VGG_net[1]
        self.relu1 = VGG_net[2]
        self.conv1_2 = VGG_net[3]
        self.bn2 = VGG_net[4]
        self.relu2 = VGG_net[5]
        self.maxpool1 = VGG_net[6]

        self.conv2_1 = VGG_net[7]
        self.bn3 = VGG_net[8]
        self.relu3 = VGG_net[9]
        self.conv2_2 = VGG_net[10]
        self.bn4 = VGG_net[11]
        self.relu4 = VGG_net[12]
        self.maxpool2 = VGG_net[13]

        self.conv3_1 = VGG_net[14]
        self.bn5 = VGG_net[15]
        self.relu5 = VGG_net[16]
        self.conv3_2 = VGG_net[17]
        self.bn6 = VGG_net[18]
        self.relu6 = VGG_net[19]
        self.conv3_3 = VGG_net[20]
        self.bn7 = VGG_net[21]
        self.relu7 = VGG_net[22]
        self.maxpool3 = VGG_net[23]

        self.conv4_1 = VGG_net[24]
        self.bn8 = VGG_net[25]
        self.relu8 = VGG_net[26]
        self.conv4_2 = VGG_net[27]
        self.bn9 = VGG_net[28]
        self.relu9 = VGG_net[29]
        self.conv4_3 = VGG_net[30]
        self.bn10 = VGG_net[31]
        self.relu10 = VGG_net[32]
        self.maxpool4 = VGG_net[33]

        self.conv5_1 = VGG_net[34]
        self.bn11 = VGG_net[35]
        self.relu11 = VGG_net[36]
        self.conv5_2 = VGG_net[37]
        self.bn12 = VGG_net[38]
        self.relu12 = VGG_net[39]
        self.conv5_3 = VGG_net[40]
        self.bn13 = VGG_net[41]
        self.relu13 = VGG_net[42]
        self.maxpool5 = VGG_net[43]

        self.conv_fc1 = nn.Conv2d(512,1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_fc1 = nn.BatchNorm2d(1024)
        # self.drop1 = nn.Dropout(p=0.5)
        self.conv_fc2 = nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_fc2 = nn.BatchNorm2d(1024)
        # self.drop2 = nn.Dropout(p=0.5)

        self.upsample1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2,bias=False)

        self.deconv1_1 = nn.Conv2d(1024,512,kernel_size=3, stride=1, padding=1, bias=False)
        self.debn1_1 = nn.BatchNorm2d(512)
        self.deconv1_2 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False)
        self.debn1_2 = nn.BatchNorm2d(512)

        self.upsample2 = nn.ConvTranspose2d(512,512,kernel_size=2,stride=2,bias=False)

        self.deconv2_1 = nn.Conv2d(1024,512, kernel_size=3,stride=1, padding=1, bias=False)
        self.debn2_1 = nn.BatchNorm2d(512)
        self.deconv2_2 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,bias=False)
        self.debn2_2 = nn.BatchNorm2d(512)

        self.upsample3 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,bias=False)

        self.deconv3_1 = nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1,bias=False)
        self.debn3_1 = nn.BatchNorm2d(256)
        self.deconv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.debn3_2 = nn.BatchNorm2d(256)

        self.upsample4 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,bias=False)

        self.deconv4_1 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1,bias=False)
        self.debn4_1 = nn.BatchNorm2d(128)
        self.deconv4_2 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=False)
        self.debn4_2 = nn.BatchNorm2d(128)

        self.upsample5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)

        self.deconv5_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.debn5_1 = nn.BatchNorm2d(64)
        self.deconv5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.debn5_2 = nn.BatchNorm2d(64)

        self.deconv_final = nn.Conv2d(64,21,kernel_size=1,stride=1,bias=False)

    def forward(self,x):
        # out = self.VGG_net(x)
        out = self.relu1(self.bn1(self.conv1_1(x)))
        out = self.relu2(self.bn2(self.conv1_2(out)))
        out1 = out #256, 64
        out = self.maxpool1(out) #128, 64

        out = self.relu3(self.bn3(self.conv2_1(out)))
        out = self.relu4(self.bn4(self.conv2_2(out)))
        out2 = out #128, 128
        out = self.maxpool2(out) #64, 128

        out = self.relu5(self.bn5(self.conv3_1(out)))
        out = self.relu6(self.bn6(self.conv3_2(out)))
        out = self.relu7(self.bn7(self.conv3_3(out)))
        out3 = out #64, 256
        out = self.maxpool3(out) #32, 256

        out = self.relu8(self.bn8(self.conv4_1(out)))
        out = self.relu9(self.bn9(self.conv4_2(out)))
        out = self.relu10(self.bn10(self.conv4_3(out)))
        out4 = out #32, 512
        out = self.maxpool4(out) #16, 512

        out = self.relu11(self.bn11(self.conv5_1(out)))
        out = self.relu12(self.bn12(self.conv5_2(out)))
        out = self.relu13(self.bn13(self.conv5_3(out)))
        out5 = out  # 16, 512
        out = self.maxpool4(out)  # 8, 512

        out = F.relu(self.bn_fc1(self.conv_fc1(out)))
        # out = self.drop1(out)
        out = F.relu(self.bn_fc2(self.conv_fc2(out)))
        # out = self.drop2(out)

        upsam1 = self.upsample1(out) #16, 512
        out = torch.cat((out5,upsam1), dim=1) #512,512
        out = F.relu(self.debn1_1(self.deconv1_1(out)))
        out = F.relu(self.debn1_2(self.deconv1_2(out)))

        upsam2 = self.upsample2(out) #32, 512
        out = torch.cat((out4,upsam2),dim=1) #512,512
        out = F.relu(self.debn2_1(self.deconv2_1(out)))
        out = F.relu(self.debn2_2(self.deconv2_2(out)))

        upsam3 = self.upsample3(out) #64, 256
        out = torch.cat((out3,upsam3), dim=1) #256,256
        out = F.relu(self.debn3_1(self.deconv3_1(out)))
        out = F.relu(self.debn3_2(self.deconv3_2(out)))

        upsam4 = self.upsample4(out) #128, 128
        out = torch.cat((out2,upsam4), dim=1) #128,128
        out = F.relu(self.debn4_1(self.deconv4_1(out)))
        out = F.relu(self.debn4_2(self.deconv4_2(out)))

        upsam5 = self.upsample5(out) #256, 64
        out = torch.cat((out1, upsam5), dim=1) #64,64
        out = F.relu(self.debn5_1(self.deconv5_1(out)))
        out = F.relu(self.debn5_2(self.deconv5_2(out)))

        out = self.deconv_final(out)

        return out