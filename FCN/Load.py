import os
import cv2
import numpy as np
import torch
from random import shuffle



def data_loader(train_file, test_file):
    img_path = "\VOCtrainval_11-May-2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages" # Enter the path to the image folder
    gt_img_path = "\VOCtrainval_11-May-2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClassAug" # Enter the path to the gt folder
    train_imgs_path, train_gts_path = [], []
    train_data = np.zeros((10582, 256, 256, 3), dtype=np.float32)
    train_gt_arr = np.zeros((10582, 256, 256), dtype=np.long)
    test_img_arr = np.zeros((1449, 256, 256, 3), dtype=np.float32)
    test_gt_arr = np.zeros((1449, 256, 256), dtype=np.long)

    with open(train_file, 'r') as img_name:
        train_img_name = []
        while True:
            names = img_name.readline().strip()
            if not names:
                break
            train_img_name.append(names)
            train_imgs_path.append(os.path.join(img_path,names+'.jpg'))
            train_gts_path.append(os.path.join(gt_img_path,names+'.png'))

        for path in range(10582):
            tr_img = cv2.imread(train_imgs_path[path]) / 255 * 2.0 - 1.0
            gt_img = cv2.imread(train_gts_path[path])
            train_resize_img = cv2.resize(tr_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            train_resize_gt = cv2.resize(gt_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            train_data[path] = train_resize_img

            h, w, _ = train_resize_gt.shape
            for a in range(h):
                for b in range(w):
                    if train_resize_gt[a, b, 0] == 255:
                        train_resize_gt[a, b, 0] = 0

            train_gt_arr[path] = train_resize_gt[:, :, 0]
    img_name.close()

    with open(test_file, 'r') as img_name:
        q = 0
        test_img_name = []
        while True:
            names = img_name.readline().strip()
            if not names:
                break
            test_img_name.append(names)
            img = cv2.imread(os.path.join(img_path, names+'.jpg')) / 255 * 2.0 - 1.0
            gt = cv2.imread(os.path.join(gt_img_path, names + '.png'))[:,:,0]

            test_img_arr[q] = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)
            test_gt_arr[q] = cv2.resize(gt, (256,256), interpolation=cv2.INTER_NEAREST)

            # h,w = test_resize_gt.shape
            # for i in range(h):
            #     for j in range(w):
            #         # if test_resize_gt[i,j] == 255:
            #         #     test_resize_gt[i,j] = 0
            q += 1 #1449
    img_name.close()
    return train_data, train_gt_arr, test_img_arr, test_gt_arr, train_img_name, test_img_name

def shuffle_Data(image, gt):
    shuffle_list = (list(range(10582)))
    shuffle(shuffle_list)
    # shuffle_img = np.zeros((10582, 256, 256, 3), dtype=np.float32)
    # shuffle_gt = np.zeros((10582, 256, 256), dtype=np.long)
    train_img = np.zeros((10582, 256, 256, 3), dtype=np.float32)
    train_gt = np.zeros((10582, 256, 256), dtype=np.uint8)

    for i in range(10582):
        train_resize_img = image[shuffle_list[i]]
        train_resize_gt = gt[shuffle_list[i]]

        if np.random.randint(2):
            train_resize_img = cv2.flip(train_resize_img, 1)
            train_resize_gt = cv2.flip(train_resize_gt, 1)
        if np.random.randint(2):
            h, w, _ = train_resize_img.shape
            x_start = np.random.randint(0, 21)
            y_start = np.random.randint(0, 21)
            # tr_img_crop = tr_img_pad[x_start:x_start + 256, y_start:y_start + 256, :]
            # gt_img_crop = gt_img_pad[x_start:x_start + 256, y_start:y_start + 256, :]
            tr_img_crop = train_resize_img[x_start:x_start + 236, y_start:y_start + 236, :]
            gt_img_crop = train_resize_gt[x_start:x_start + 236, y_start:y_start + 236]
            train_resize_img = cv2.resize(tr_img_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            train_resize_gt = cv2.resize(gt_img_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        train_img[i] = train_resize_img
        train_gt[i] = train_resize_gt
    return train_img, train_gt