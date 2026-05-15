import cv2
import os
from Load import data_loader, shuffle_Data
from network import UNet
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchvision.models as models

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
train_name_file = r'\VOCtrainval_11-May-2012\VOCtrainval_11-May-2012\VOCdevkit\train_aug.txt' # Provide the TXT file path for your train image list.
test_name_file = r"\VOCtrainval_11-May-2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt" # Provide the TXT file path for your test image list.

train_imgs, train_gts, test_imgs, test_gts, train_img_name, test_img_name = data_loader(train_name_file, test_name_file)
print('data load done')


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

model_vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features

model = UNet(model_vgg).to(DEVICE)
EPOCHS = 500
batch_size = 32
exp_name = 'UNet_accuracy'
save_t = 10
# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.0001)
optimizer = optim.Adam(model.parameters(), lr = 0.00001, weight_decay=0.0001)

def train(MODEL, train_img, train_gt, batch_size, optimizer, device):
    MODEL.train()
    i, train_loss = 0, 0
    # train_IOU = np.zeros((21, 21), dtype=np.int32)
    shuffle_img, shuffle_gt = shuffle_Data(train_img, train_gt)
    for batch_idx in tqdm(range(int(len(train_img) / batch_size))):
        bth_gts = torch.from_numpy(np.array((shuffle_gt[i:i + batch_size, :, :]), dtype=np.int64))
        batch_img = torch.from_numpy(np.array((shuffle_img[i:i + batch_size, :, :, :]), dtype=np.float32))
        bth_img, bth_gt = batch_img.permute(0,3,1,2).to(device), bth_gts.to(device)
        optimizer.zero_grad()
        output = MODEL(bth_img)
        loss = F.cross_entropy(output, bth_gt)
        train_loss += loss
        loss.backward()
        optimizer.step()
        i += batch_size

    train_loss /= int(len(train_img) / batch_size)
    return train_loss


def evaluate(MODEL, test_img, test_gt, device, epoch):
    MODEL.eval()
    test_loss, IOUs, mIOUs, correct, each_mIOUs, cls_iou = 0, 0, 0, 0, [], []
    eval_IOUs = np.zeros((21, 21), dtype=np.int64)
    save_pred_image = np.zeros((1449,256,256), dtype=np.uint32)
    with torch.no_grad():
        for bth in tqdm(range(len(test_img))):
            eval_IOU, IOU = np.zeros((21, 21), dtype=np.int32), 0
            bth_imgs = torch.from_numpy(np.array([test_img[bth]], dtype=np.float32))
            bth_gts = torch.from_numpy(np.array([test_gt[bth]], dtype=np.int64)) #[1,256,256,21]
            bth_img, bth_gt = bth_imgs.permute(0,3,1,2).to(device), bth_gts.to(device)
            output = MODEL(bth_img)
            test_loss += F.cross_entropy(output, bth_gt, ignore_index=255).item()
            predict = torch.argmax(output, dim=1).cpu().numpy().squeeze()
            if epoch % 30 == 0:
                save_pred_image[bth,:,:] = predict
            IOU_gt = bth_gt.cpu().numpy().squeeze()
            unique_element = len(np.unique(IOU_gt))
            for i in range(256):
                for j in range(256):
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
            each_mIOUs.append(round((100. * IOU / unique_element).item(), 2))
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
    test_loss /= len(test_img)
    return test_loss, mIOUs, each_mIOUs, save_pred_image, cls_iou


def pred_color_img(gt_pred_image, gt_image_name, mIOU, epoch):
    test_path = './test_savePredict_final'
    gt_pred_rgb = np.zeros((1449,256,256,3), dtype=np.uint8)

    for idx2,img2 in enumerate(gt_pred_image):
        for i in range(256):
            for j in range(256):
                num = str(img2[i][j])
                gt_pred_rgb[idx2,i,j,:] = class_colormap[num]
        os.makedirs(f"{test_path}/{epoch}", exist_ok=True)
        cv2.imwrite(f"{test_path}/{epoch}/({mIOU[idx2]})_{gt_image_name[idx2]}.png", gt_pred_rgb[idx2].astype(np.uint8))


for epoch in range(1,EPOCHS+1):
    if epoch % 30 == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    check_lr = optimizer.param_groups[0]['lr']
    train_loss = train(model, train_imgs, train_gts, batch_size, optimizer, DEVICE)
    print("train done")
    test_loss, test_miou, each_mIOU, pred_img, cls_IOU = evaluate(model, test_imgs, test_gts, DEVICE, epoch)
    print("test done")
    if epoch % 30 == 0:
        pred_color_img(pred_img, test_img_name, each_mIOU, epoch)
        print('img save done')
    with open(f'./model_accuracy/{exp_name}.txt' , 'a') as f:
        f.write('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test mIOU : {:.3f}%, LR : {:.5f}\n'.format(epoch, train_loss, test_loss, test_miou, check_lr))
    print('[{}] Train Loss : {:.4f}, Test Loss : {:.4f}, mIOU : {:.5f}%, Lr : {:.5f}'.format(epoch, train_loss, test_loss, test_miou, check_lr))
    if epoch % save_t == 0:
        torch.save(model.state_dict(), f'./model_save/{epoch}_model.pt')