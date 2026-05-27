import torch.nn as nn
import cv2
from load import mini_batch, data_loader
from network import Classifier, Load_backBone
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4096
EPOCHS = 300
save_t = 10
iteration = 500
restore = False
exp_name = 'DINOv2_accuracy'

#데이터 불러오기
train_list_path = '/dataset/Tiny_ImageNet/train_img_dir.txt'
test_list_path = '/dataset/Tiny_ImageNet/test_img_dir.txt'
train_img, train_img_gt, test_img, test_img_gt, test_img_name = data_loader(train_list_path, test_list_path)
print('finish load data')
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')


dinov2.patch_embed.proj.kernel_size = 16
dinov2.patch_embed.proj.stride = 16
dinov2.patch_embed.img_size = 128
dinov2.patch_embed.patch_size = (16,16)


# pre-trianed position embedding interpolation
trained_pos_emb = dinov2.pos_embed[:,1:,:] #[1,196,768]
H = W = int(trained_pos_emb.shape[1]**0.5)
devide_pos_emb = trained_pos_emb.reshape(1,H,W,768).permute(0,3,1,2)
inter_pos_emb = nn.functional.interpolate(devide_pos_emb, size=(8,8), mode='nearest')
new_pos_emb = inter_pos_emb.permute(0,3,1,2).reshape(1,-1,768)
new_pos_emb = torch.cat([dinov2.pos_embed[:,0,:].reshape(1,1,768), new_pos_emb], dim=1)
dinov2.pos_embed = nn.Parameter(new_pos_emb)

# ViT_H = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT).to(DEVICE)

# pre-trianed position embedding interpolation
# trained_pos_emb = ViT_H.encoder.pos_embedding[:,1:,:] #[1,196,768]
# H = W = int(trained_pos_emb.shape[1]**0.5)
# devide_pos_emb = trained_pos_emb.reshape(1,H,W,1280).permute(0,3,1,2)
# inter_pos_emb = nn.functional.interpolate(devide_pos_emb, size=(6,6), mode='nearest')
# new_pos_emb = inter_pos_emb.permute(0,3,1,2).reshape(1,-1,1280)
# new_pos_emb = torch.cat([ViT_H.encoder.pos_embedding[:,0,:].reshape(1,1,1280), new_pos_emb], dim=1)
# ViT_H.encoder.pos_embedding = nn.Parameter(new_pos_emb)
# ViT_H.image_size = 128
# ViT_H.patch_size = 16
# ViT_H.heads.head.out_features = 200
# ViT_H.conv_proj = nn.Conv2d(3,1280,kernel_size=16, stride=16)

back_bone = Load_backBone().to(DEVICE)
model = Classifier().to(DEVICE)

if restore:
    model.load_state_dict(torch.load(f'{restore}_model.pth'))

optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, image_data, image_gt, batch_size, optimizer, DEVICE):
    model.train()
    train_loss, i = 0, 0
    # shuffle_img, shuffle_gt = mini_batch(image_data, image_gt, batch_size, shuffle_idx)
    for i in range(iteration):
        random_idx = np.random.randint(0,len(image_data), batch_size)
        batch_img, batch_gt = mini_batch(image_data[random_idx], image_gt[random_idx], batch_size)
        bth_img, bth_target = batch_img.to(DEVICE), batch_gt.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            embedding = back_bone(bth_img)
        output = model(embedding)
        loss = F.cross_entropy(output, bth_target)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= len(image_data)
    return train_loss

def evaluate(model, test_data, test_gt, DEVICE):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for bth in range(len(test_data)):
            bth_test = torch.from_numpy(np.array([test_data[bth]], dtype=np.float32))/255*2.0-1.0
            bth_test_target = torch.from_numpy(np.array([test_gt[bth]], dtype=np.int64))
            bth_test, bth_test_target = bth_test.permute(0, 3, 1, 2).to(DEVICE), bth_test_target.to(DEVICE)
            embedding = back_bone(bth_test)
            output = model(embedding)
            test_loss += F.cross_entropy(output, bth_test_target, reduction='sum').item()
            predict = output.max(1, keepdim=True)[1]
            correct += predict.eq(bth_test_target.view_as(predict)).sum().item()
    test_loss /= len(test_data)
    test_accuracy = 100. * correct / len(test_data)
    return test_loss, test_accuracy

for epoch in range(1,EPOCHS+1):
    if epoch % 60 == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    check_lr = optimizer.param_groups[0]['lr']
    train_loss = train(model, train_img, train_img_gt, batch_size, optimizer, DEVICE)
    print("train done")
    test_loss, test_accuracy = evaluate(model, test_img, test_img_gt, DEVICE)
    os.makedirs('./model_accuracy', exist_ok=True)
    with open(f'./model_accuracy/{exp_name}.txt' , 'a') as f:
        # if epoch == 1:
        #     f.write(f'patch size : {patch_size}, hidden dim : {hidden_dim}, heads : {heads}, mlp_dim : {mlp_dim}, Layers : {layers}\n')
        f.write('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}\n'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
    print("test done")
    print('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
    if epoch % save_t == 0:
        os.makedirs('./model_save', exist_ok=True)
        torch.save(model.state_dict(), f'./model_save/{epoch}_pretrained_model.pt')