import os
import numpy as np
import cv2
import torch
from random import shuffle

def data_loader(train_list_path, test_list_path):

    # train_images_arr = []
    train_images = np.zeros((207005, 128, 128, 3), dtype=np.uint8)
    train_images_gt = np.zeros(207005, dtype=np.uint8)
    test_images = np.zeros((51752, 128, 128, 3), dtype=np.uint8)
    test_images_gt = np.zeros(51752, dtype=np.uint8)
    test_image_name = []

    with open(train_list_path, 'r') as train_list:
        i = 0
        while True:
            image = train_list.readline().strip()
            if not image:
                break
            img = cv2.imread(image)
            train_images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            split_train_dir = int(image.split('/')[7])
            train_images_gt[i] = split_train_dir
            # train_images_arr.append(image)
            i += 1
    train_list.close()

    with open(test_list_path, 'r') as test_list:
        j = 0
        while True:
            label = test_list.readline().strip()
            if not label:
                break
            labels = cv2.imread(label)
            lb_split = label.split('/')
            split_test_dir = int(lb_split[7])
            img_name = lb_split[8]
            test_image_name.append(img_name)
            test_images_gt[j] = split_test_dir
            test_images[j] = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            j += 1
    test_list.close()

    return train_images, train_images_gt, test_images, test_images_gt, test_image_name

def checknum(dirname):
    path = f"/dataset/Tiny_ImageNet/{dirname}/"

    cls_list = os.listdir(path)
    q = 0
    for i in cls_list:
        img_dir = os.path.join(path,i)
        img_list = os.listdir(img_dir)
        for j in img_list:
            q += 1
    print(q)


def mini_batch(data, gt, batch):
    imgs = torch.zeros((batch, 128, 128, 3), dtype=torch.float32)
    target = torch.zeros(batch, dtype=torch.int64)

    for idx,i in enumerate(data):
        img = data[idx] / 255 * 2.0 - 1.0
        shuffle_gt = gt[idx]
        if np.random.randint(2):
            img = cv2.flip(img, 1)
        if np.random.randint(2):
            #zero padding
            h, w, _ = img.shape
            padding_space = np.zeros((h+20, w+20, 3), dtype=img.dtype)
            padding_space[10:10+h, 10:10+w, :] = img
            img = padding_space
            #random Crop
            x_start = np.random.randint(0,21)
            y_start = np.random.randint(0,21)
            cropped_img = img[x_start:x_start+128, y_start:y_start+128, :]
            img = cropped_img

            #PCA_img = img.reshape(-1, 3)
        target[idx] = shuffle_gt
        imgs[idx] = torch.tensor(img, dtype=torch.float32)
    imgs = torch.permute(imgs, (0, 3, 1, 2))

    return imgs, target

