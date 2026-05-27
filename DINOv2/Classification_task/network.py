import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

def Load_backBone():

    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    # dinov2.patch_embed.proj.kernel_size = 16
    # dinov2.patch_embed.proj.stride = 16
    # dinov2.patch_embed.img_size = 128
    # dinov2.patch_embed.patch_size = (16, 16)

    # pre-trianed position embedding interpolation
    # trained_pos_emb = dinov2.pos_embed[:, 1:, :]  # [1,196,768]
    # H = W = int(trained_pos_emb.shape[1] ** 0.5)
    # devide_pos_emb = trained_pos_emb.reshape(1, H, W, 384).permute(0, 3, 1, 2)
    # inter_pos_emb = nn.functional.interpolate(devide_pos_emb, size=(8, 8), mode='nearest')
    # new_pos_emb = inter_pos_emb.permute(0, 3, 1, 2).reshape(1, -1, 384)
    # temp = 0
    # new_pos_emb = torch.cat([dinov2.pos_embed[:, 0, :].reshape(1, 1, 384), new_pos_emb], dim=1)
    # dinov2.pos_embed = nn.Parameter(new_pos_emb)

    dinov2.eval()
    return dinov2


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        # self.fc1 = nn.Linear(768, 200)
        self.fc1 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 200),
        )
    # def student_imgs(self,x):
    #     rand_x = torch.randint(low=0,high=32, size=(2, 1))
    #     rand_y = torch.randint(low=0, high=32, size=(2, 1))
    #     global_crop_img1 = x[:, :, rand_x[0]:rand_x[0] + 96, rand_y[0]:rand_y[0] + 96]
    #     global_crop_img2 = x[:, :, rand_x[1]:rand_x[1] + 96, rand_y[1]:rand_y[1] + 96]
    #     # local_crop_img1 = x[:, :, rand_x[0]:rand_x[0] + 48, rand_y[0]:rand_y[0] + 48]
    #     # local_crop_img2 = x[:, :, rand_x[1]:rand_x[1] + 48, rand_y[1]:rand_y[1] + 48]
    #     # local_crop_img3 = x[:, :, rand_x[2]:rand_x[2] + 48, rand_y[2]:rand_y[2] + 48]
    #     # local_crop_img4 = x[:, :, rand_x[3]:rand_x[3] + 48, rand_y[3]:rand_y[3] + 48]
    #     # local_crop_img5 = x[:, :, rand_x[4]:rand_x[4] + 48, rand_y[4]:rand_y[4] + 48]
    #     # local_crop_img6 = x[:, :, rand_x[5]:rand_x[5] + 48, rand_y[5]:rand_y[5] + 48]
    #     crop_imgs = [global_crop_img1, global_crop_img2]
    #     return crop_imgs

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Seg_Classifier(nn.Module):
    def __init__(self):
        super(Seg_Classifier,self).__init__()


    def forward(self,x):

        return x