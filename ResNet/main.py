import os
from Load import mini_batch, data_loader
from CBAM_network import ResNet101
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image

batch_size = 256
EPOCHS = 150
save_t = 10
restore = False
exp_name = '50_layer_accuracy'
#데이터 불러오기

train_list_path = '/home/aivs/바탕화면/adj_test/DenseNet/train_img_dir.txt'
test_list_path = '/home/aivs/바탕화면/adj_test/DenseNet/test_img_dir.txt'
train_img_arr, train_img_gt, test_img, test_img_gt, test_img_name = data_loader(train_list_path, test_list_path)

print('data load finish')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

model = ResNet101().to(DEVICE)
model.load_state_dict(torch.load('/home/aivs/바탕화면/adj_test/ResNet/CBAM_save/90_model.pt', map_location=DEVICE))
if restore:
    model.load_state_dict(torch.load(f'{restore}_model.pth'))

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
#scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1)

# def train(model, image_data, image_gt, batch_size, optimizer, DEVICE):
#     model.train()
#     train_loss = 0
#     for batch_idx in range(int(len(image_data) / batch_size)):
#         batch_img, batch_target = mini_batch(image_data, image_gt, batch_size)
#         bth_img, bth_target = batch_img.to(DEVICE), batch_target.to(DEVICE)
#         optimizer.zero_grad()
#         output = model(bth_img)
#         loss = F.cross_entropy(output, bth_target)
#         train_loss += loss
#         loss.backward()
#         optimizer.step()
#     train_loss /= len(image_data)
#     return train_loss
#
# def evaluate(model,test_data, test_gt, DEVICE):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for bth in range(len(test_data)):
#             #batch_val, batch_val_target = mini_batch(test_data, test_gt, bth_size)
#             test_img, test_target = test_data[bth], test_gt[bth]
#             # bth_test, bth_test_target = torch.from_numpy(test_img.astype(np.float32)), torch.from_numpy(test_target.astype(np.float32))
#             bth_test = torch.from_numpy(np.array([test_data[bth]], dtype=np.float32))/255*2.0-1.0
#             bth_test_target = torch.from_numpy(np.array([test_gt[bth]], dtype=np.int64))
#             bth_test, bth_test_target = bth_test.permute(0, 3, 1, 2).to(DEVICE), bth_test_target.to(DEVICE)
#             output = model(bth_test)
#             test_loss += F.cross_entropy(output, bth_test_target, reduction='sum').item()
#             predict = output.max(1, keepdim=True)[1]
#             #print(predict)
#             correct += predict.eq(bth_test_target.view_as(predict)).sum().item()
#
#     test_loss /= len(test_data)
#     test_accuracy = 100. * correct / len(test_data)
#     return test_loss, test_accuracy
def train(model, image_data, image_gt, batch_size, optimizer, DEVICE):
    model.train()
    image_gt = torch.tensor(image_gt, dtype=torch.int32)
    batch_split_img, batch_split_gt = [], []
    batch_split_img = np.array_split(image_data, math.ceil(len(image_data) / batch_size))
    batch_split_gt = np.array_split(image_gt, math.ceil(len(image_data) / batch_size))
    train_loss = 0
    for batch_idx in range(int(len(image_data) / batch_size)):
        batch_img, batch_target = mini_batch(batch_split_img[batch_idx], batch_split_gt[batch_idx], batch_size)
        bth_img, bth_target = batch_img.to(DEVICE), batch_target.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            output,_ = model(bth_img)
        loss = F.cross_entropy(output, bth_target)
        train_loss += loss
        #loss.backward()
        optimizer.step()
    train_loss /= len(image_data)
    return train_loss

def evaluate(model, test_data, test_gt, DEVICE, img_name, heat):
    imgs = torch.zeros((len(test_data), 128, 128, 3), dtype=torch.float32)
    target = torch.zeros(len(test_data), dtype=torch.int64)
    model.eval()
    test_loss = 0
    correct = 0
    q= 0

    with torch.no_grad():
        for bth in range(len(test_data)):
            #print(test_data[bth].shape)

            bth_test = torch.from_numpy(np.array([test_data[bth]], dtype=np.float32))/255*2.0-1.0
            bth_test_target = torch.from_numpy(np.array([test_gt[bth]], dtype=np.int64))
            bth_test, bth_test_target = bth_test.permute(0, 3, 1, 2).to(DEVICE), bth_test_target.to(DEVICE)
            output,img = model(bth_test)
            #print(test_gt[bth])
            if heat == True:
                #heatmap
                img = np.squeeze(img)
                img = torch.mean(img, dim=0)
                img = ((img - img.mean()) / (img.max() - img.mean()) * 255)
                img = img.to('cpu').numpy().astype(dtype=np.uint8)
                img = cv2.resize(img,(128,128), interpolation=cv2.INTER_LINEAR) #(128, 128)
                heatmap = plt.cm.jet(np.array(img) / 255.)
                heatmap[:,:,:3] = (heatmap[:,:,:3] * 255)
                heatmap = heatmap.astype(np.uint8)
                heatmap = Image.fromarray(heatmap).convert('RGBA')
                org_img = Image.fromarray(test_data[bth])
                heatmap.putalpha(96)
                image = Image.alpha_composite(org_img.convert('RGBA'), heatmap)
                op_lr = optimizer.param_groups[0]['lr']
                os.makedirs(f'./heatmap/{op_lr}/{test_gt[bth]}', exist_ok=True)
                image.save(f'./heatmap/{op_lr}/{test_gt[bth]}/{img_name[q]}','PNG')
                q += 1

            test_loss += F.cross_entropy(output, bth_test_target, reduction='sum').item()
            predict = output.max(1, keepdim=True)[1]
            #print(predict)
            correct += predict.eq(bth_test_target.view_as(predict)).sum().item()

    test_loss /= len(test_data)
    test_accuracy = 100. * correct / len(test_data)
    return test_loss, test_accuracy

for epoch in range(1,EPOCHS+1):
    optim_lr = optimizer.param_groups[0]['lr']
    if epoch % 30 == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    check_lr = optimizer.param_groups[0]['lr']
    if optim_lr != check_lr or epoch == 1:
        heatM = True
    else:
        heatM = False
    epochs = epoch * 10
    #model.load_state_dict(torch.load(f'/home/aivs/바탕화면/adj_test/ResNet/model_save/{epochs}_model.pt'))
    train_loss = train(model, train_img_arr, train_img_gt, batch_size, optimizer, DEVICE)
    print("train done")
    test_loss, test_accuracy = evaluate(model, test_img, test_img_gt, DEVICE, test_img_name, heatM)
    #scheduler.step(test_loss)
    with open(f'./model_accuracy/{exp_name}.txt' , 'a') as f:
        f.write('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}\n'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
    print("test done")
    print('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
    if epoch % save_t == 0:
        torch.save(model.state_dict(), f'./CBAM_save/{epoch}_model.pt')

# for i in range(1, 21):
#     n = i*10
#     #model.load_state_dict(torch.load(f'./model_save/{n}_model.pt'))
#     test_loss, test_accuracy = evaluate(model, test_data, test_gt, DEVICE)
#     with open(f'./model_accuracy/{exp_name}.txt' , 'a') as f:
#         f.write(f'Epoch: {i} >> Acc: {test_accuracy}\n')
#     print("test done")
#     print('[{}] Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(i, test_loss, test_accuracy))
#     if epoch % save_t == 0:
#         torch.save(model.state_dict(), f'./model_save/{epoch}_model.pt')


# def visualize_heatmap(pil_image, heatmap, bbox=None):
#     if isinstance(heatmap, torch.Tensor):
#         heatmap = heatmap.detach().cpu().numpy()
#     heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
#     heatmap = plt.cm.jet(np.array(heatmap) / 255.)
#     heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
#     heatmap = Image.fromarray(heatmap).convert("RGBA")
#     heatmap.putalpha(128)
#     overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)
#
#     if bbox is not None:
#         width, height = pil_image.size
#         xmin, ymin, xmax, ymax = bbox
#         draw = ImageDraw.Draw(overlay_image)
#         draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="green", width=3)
#     return overlay_image