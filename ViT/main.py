import torch.nn as nn

from Load import mini_batch, data_loader
from network import ViT
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from random import shuffle
import torchvision.models as models
from einops import rearrange

batch_size = 256
EPOCHS = 300
save_t = 10
restore = False
exp_name = 'ViT_accuracy'

#데이터 불러오기
train_list_path = '/dataset/Tiny_ImageNet/train_img_dir.txt' # Provide the TXT file path for your train image list.
test_list_path = '/dataset/Tiny_ImageNet/test_img_dir.txt' # Provide the TXT file path for your test image list.
train_img, train_img_gt, test_img, test_img_gt, test_img_name = data_loader(train_list_path, test_list_path)
print('finish load data')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_ViT = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(DEVICE)

# model_ViT.heads.head.out_features = 200
# model_ViT.image_size = 128

# pre-trianed position embedding interpolation
# trained_pos_emb = model_ViT.encoder.pos_embedding[:,1:,:] #[1,196,768]
# H = W = int(trained_pos_emb.shape[1]**0.5)
# devide_pos_emb = trained_pos_emb.reshape(1,H,W,768).permute(0,3,1,2)
# inter_pos_emb = nn.functional.interpolate(devide_pos_emb, size=(8,8), mode='nearest')
# new_pos_emb = inter_pos_emb.permute(0,3,1,2).reshape(1,-1,768)
# new_pos_emb = torch.cat([model_ViT.encoder.pos_embedding[:,0,:].reshape(1,1,768), new_pos_emb], dim=1)
# model_ViT.encoder.pos_embedding = nn.Parameter(new_pos_emb)

image_size = 128
patch_size = 16
hidden_dim = 768
heads = 12
mlp_dim = 3072
layers = 12
model = ViT(image_size=image_size,patch_size=patch_size, hidden_dim=hidden_dim,
            heads=heads, mlp_dim=mlp_dim, layers=layers, dropout=0.1).to(DEVICE)

# model = model_ViT
if restore:
    model.load_state_dict(torch.load(f'{restore}_model.pth'))

optimizer = optim.SGD(model.parameters(), lr=0.3)
shuffle_idx = list(range(len(train_img)))

def train(model, image_data, image_gt, batch_size, optimizer, DEVICE):
    model.train()
    shuffle(shuffle_idx)
    # image_gt = torch.tensor(image_gt, dtype=torch.uint8)
    train_loss, i = 0, 0
    # shuffle_img, shuffle_gt = mini_batch(image_data, image_gt, batch_size, shuffle_idx)
    for batch_idx in range(len(image_data) // batch_size):
        batch_img, batch_gt = mini_batch(image_data[shuffle_idx[i:i+batch_size],:,:,:], image_gt[shuffle_idx[i:i+batch_size]], batch_size)
        batch_gt = batch_gt
        bth_img, bth_target = batch_img.to(DEVICE), batch_gt.to(DEVICE)
        optimizer.zero_grad()
        output, _, _, _ = model(bth_img)
        loss = F.cross_entropy(output, bth_target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        i += batch_size
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
            output = model(bth_test)
            test_loss += F.cross_entropy(output, bth_test_target, reduction='sum').item()
            predict = output.max(1, keepdim=True)[1]
            correct += predict.eq(bth_test_target.view_as(predict)).sum().item()
    test_loss /= len(test_data)
    test_accuracy = 100. * correct / len(test_data)
    return test_loss, test_accuracy

for epoch in range(1,EPOCHS+1):
    if epoch % 50 == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    check_lr = optimizer.param_groups[0]['lr']
    train_loss = train(model, train_img, train_img_gt, batch_size, optimizer, DEVICE)
    print("train done")
    test_loss, test_accuracy = evaluate(model, test_img, test_img_gt, DEVICE)
    with open(f'./model_accuracy/{exp_name}.txt' , 'a') as f:
        if epoch == 1:
            f.write(f'patch size : {patch_size}, hidden dim : {hidden_dim}, heads : {heads}, mlp_dim : {mlp_dim}, Layers : {layers}\n')
        f.write('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}\n'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
    print("test done")
    print('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
    if epoch % save_t == 0:
        torch.save(model.state_dict(), f'./model_save/{epoch}_pretrained_model.pt')