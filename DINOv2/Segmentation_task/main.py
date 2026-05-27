import torch.nn as nn

from load import mini_batch, data_loader
from network import Seg_Classifier, load_backbone
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import cv2
from random import shuffle
from torchvision import models
from einops.layers.torch import Rearrange
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_colormap = {
    "0":[0,0,0], #back ground
    "1":[0,0,128], #aero plane
    "2":[0,128,0], #Bicycle
    "3":[0,128,128], # Bird
    "4":[128,0,0], #Boat
    "5":[128,0,128], #Bottle
    "6":[128,128,0], #Bus
    "7":[128,128,128], #Car
    "8":[0,0,64], #Cat
    "9":[0,0,192], #Chair
    "10":[0,128,64], #Cow
    "11":[0,128,192], #Dining Table
    "12":[128,0,64], #Dog
    "13":[128,0,192], #Horse
    "14":[128,128,64], #Motorbike
    "15":[128,128,192], #Person
    "16":[0,64,0], #Potted Plant
    "17":[0,64,128], #Sheep
    "18":[0,192,0], #Sofa
    "19":[0,192,128], #Train
    "20":[128,64,0], #TV/Monitor
}
image_size = 266
patch_size = 14
batch_size = 64
EPOCHS = 300
save_t = 10
iteration = 500
restore = False
exp_name = 'DINOv2_accuracy'

#데이터 불러오기
train_name_file = '/Desktop/hdd/dataset/VOCtrainval_11-May-2012/VOCdevkit/train_aug.txt'
test_name_file = '/Desktop/hdd/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
train_img, train_img_gt, test_img, test_img_gt,train_img_name, test_img_name = data_loader(train_name_file, test_name_file)
print('finish load data')
hidden_dim = 768
out_dim = (image_size // patch_size)**2 * 3

back_bone = load_backbone().to(DEVICE)


model = Seg_Classifier(hidden_dim, out_dim).to(DEVICE)
if restore:
    model.load_state_dict(torch.load(f'{restore}_model.pth'))

optimizer = optim.Adam(model.parameters(), lr=0.01)
def train(model, image_data, image_gt, batch_size, optimizer, DEVICE):
    model.train()
    train_loss, i = 0, 0
    # shuffle_img, shuffle_gt = mini_batch(image_data, image_gt, batch_size, shuffle_idx)
    for _ in range(iteration):
        random_idx = np.random.randint(0,len(image_data), batch_size)
        batch_img, batch_gt = mini_batch(image_data[random_idx], image_gt[random_idx])
        bth_img, bth_target = batch_img.permute(0,3,1,2).to(DEVICE), batch_gt.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            # embedding = back_bone(bth_img)
            features = back_bone.forward_features(bth_img)
            embedding = features['x_norm_patchtokens']
        output = model(embedding)
        loss = F.cross_entropy(output, bth_target, ignore_index=255)
        train_loss += loss
        loss.backward()
        optimizer.step()
        i += batch_size
    # scheduler.step()
    train_loss /= len(image_data)
    return train_loss

def evaluate(model, test_data, test_gt, DEVICE, epoch):
    model.eval()
    test_loss, IOUs, mIOUs, correct, each_mIOUs, cls_iou = 0, 0, 0, 0, [], []
    eval_IOUs = np.zeros((21, 21), dtype=np.int64)
    save_pred_image = np.zeros((1449,266,266), dtype=np.uint32)
    with torch.no_grad():
        for bth in range(len(test_data)):
            eval_IOU, IOU = np.zeros((21, 21), dtype=np.int32), 0
            bth_test = torch.from_numpy(np.array([test_data[bth]], dtype=np.float32))/255*2.0-1.0
            bth_test_target = torch.from_numpy(np.array([test_gt[bth]], dtype=np.int64))
            bth_test, bth_test_target = bth_test.permute(0, 3, 1, 2).to(DEVICE), bth_test_target.to(DEVICE)
            # embedding = back_bone(bth_test)
            features = back_bone.forward_features(bth_test)
            embedding = features['x_norm_patchtokens']

            output = model(embedding)
            test_loss += F.cross_entropy(output, bth_test_target, ignore_index=255).item()
            predict = torch.argmax(output, dim=1).cpu().numpy().squeeze()
            if epoch % 1 == 0:
                save_pred_image[bth,:,:] = predict
            IOU_gt = bth_test_target.cpu().numpy().squeeze()
            unique_element = np.unique(IOU_gt)
            len_unique_element = len(unique_element[unique_element != 255])
            for i in range(266):
                for j in range(266):
                    if IOU_gt[i][j] == 255:
                        continue
                    else:
                        eval_IOUs[IOU_gt[i][j]][predict[i][j]] += 1
                        eval_IOU[IOU_gt[i][j]][predict[i][j]] += 1
            w_sum, h_sum = np.sum(eval_IOU, axis=1), np.sum(eval_IOU, axis=0)
            for q in range(21):
                denominator = (w_sum[q] + h_sum[q] - eval_IOU[q][q])
                if denominator == 0:
                    IOU += 0
                else:
                    IOU += (eval_IOU[q][q] / denominator)
            each_mIOUs.append(round((100. * IOU / len_unique_element).item(), 2))
    width_sum, height_sum = np.sum(eval_IOUs, axis=1), np.sum(eval_IOUs, axis=0)
    for q in range(21):
        denominator2 = (width_sum[q] + height_sum[q] - eval_IOUs[q][q])
        if denominator2 == 0:
            cls_iou.append(0)
        else:
            IOU = (eval_IOUs[q][q] / denominator2)
            cls_iou.append(round(100. * IOU.item(), 2))
    IOUs = np.sum(cls_iou)
    mIOUs = IOUs / 21
    test_loss /= len(test_data)
    return test_loss, mIOUs, each_mIOUs, save_pred_image, cls_iou

def pred_color_img(gt_pred_image, gt_image_name, IOU, epoch):
    test_path = './test_savePredict_Final'
    gt_pred_rgb = np.zeros((1449,266,266,3), dtype=np.uint8)

    for idx2,img2 in enumerate(gt_pred_image):
        for i in range(266):
            for j in range(266):
                num = str(img2[i][j])
                gt_pred_rgb[idx2,i,j,:] = class_colormap[num]
        os.makedirs(f"{test_path}/{epoch}", exist_ok=True)
        cv2.imwrite(f"{test_path}/{epoch}/({IOU[idx2]})_{gt_image_name[idx2]}.png", gt_pred_rgb[idx2].astype(np.uint8))


for epoch in range(1,EPOCHS+1):
    if epoch % 60 == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    check_lr = optimizer.param_groups[0]['lr']
    train_loss = train(model, train_img, train_img_gt, batch_size, optimizer, DEVICE)
    print("train done")
    test_loss, test_miou, each_mIOU, pred_img, cls_IOU = evaluate(model, test_img, test_img_gt, DEVICE, epoch)
    if epoch % 1 == 0:
        pred_color_img(pred_img, test_img_name, each_mIOU, epoch)
        print('img save done')
    os.makedirs('./model_accuracy', exist_ok=True)
    with open(f'./model_accuracy/{exp_name}.txt' , 'a') as f:
        # if epoch == 1:
        #     f.write(f'patch size : {patch_size}, hidden dim : {hidden_dim}, heads : {heads}, mlp_dim : {mlp_dim}, Layers : {layers}\n')
        f.write('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, miou : {:.3f}%, LR : {:.6f}\n'.format(epoch, train_loss, test_loss, test_miou, check_lr))
    print("test done")
    print('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, miou : {:.3f}%, LR : {:.6f}'.format(epoch, train_loss, test_loss, test_miou, check_lr))
    if epoch % save_t == 0:
        os.makedirs('./model_save', exist_ok=True)
        torch.save(model.state_dict(), f'./model_save/{epoch}_pretrained_model.pt')