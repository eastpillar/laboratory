import torch
import torch.nn as nn
import torchvision.transforms
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode


def load_backbone():
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    # dinov2.patch_embed.proj.kernel_size = 16
    # dinov2.patch_embed.proj.stride = 16
    # dinov2.patch_embed.img_size = 256
    # dinov2.patch_embed.patch_size = (16, 16)
    #
    # # pre-trianed position embedding interpolation
    # trained_pos_emb = dinov2.pos_embed[:, 1:, :]  # [1,196,768]
    # H = W = int(trained_pos_emb.shape[1] ** 0.5)
    # devide_pos_emb = trained_pos_emb.reshape(1, H, W, 768).permute(0, 3, 1, 2)
    # inter_pos_emb = nn.functional.interpolate(devide_pos_emb, size=(16, 16), mode='nearest')
    # new_pos_emb = inter_pos_emb.permute(0, 3, 1, 2).reshape(1, -1, 768)
    # new_pos_emb = torch.cat([dinov2.pos_embed[:, 0, :].reshape(1, 1, 768), new_pos_emb], dim=1)
    # dinov2.pos_embed = nn.Parameter(new_pos_emb)
    dinov2.eval()
    return dinov2

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.fc1 = nn.Linear(1024, 200)

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
        return x

class Seg_Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_token=19):
        super(Seg_Classifier,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 21),
            Rearrange('b (c1 c2) l -> b l c1 c2', c1=num_token, c2=num_token),

            # nn.ConvTranspose2d(out_dim, 21, kernel_size=14, stride=14, bias=False)
            torchvision.transforms.Resize((266,266), interpolation=InterpolationMode.BILINEAR)

        )
        # self.fc2 = nn.Linear(out_dim, 21)
    def forward(self,x):
        temp = 0
        x = self.fc1(x)
        # x = F.interpolate(x, (266,266), mode='bilinear')
        # x = self.fc2(x)
        return x