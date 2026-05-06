import os
import numpy as np
import cv2
import torch
from torchvision import transforms


def data_loader(train_list_path, test_list_path):
    # images = np.zeros((numofdata, 128, 128, 3), dtype=np.uint8)
    # gt = np.zeros(numofdata, dtype=np.uint8)

    # path = f"/home/aivs/바탕화면/dataset/Tiny_ImageNet/{dataval}/"
    train_list_path = '/home/aivs/바탕화면/dataset/Tiny_ImageNet/train_img_dir.txt'
    test_list_path = '/home/aivs/바탕화면/dataset/Tiny_ImageNet/test_img_dir.txt'
    # cls_list = os.listdir(path)

    train_images_arr = []
    train_images_gt = np.zeros(207005, dtype=np.uint8)
    test_images = np.zeros((51752, 128, 128, 3), dtype=np.uint8)
    test_images_gt = np.zeros(51752, dtype=np.uint8)

    with open(train_list_path, 'r') as train_list:
        i = 0
        while True:
            image = train_list.readline().strip()
            if not image:
                break
            # img = cv2.imread(image)
            # images[i] = img
            split_train_dir = int(image.split('/')[7])
            train_images_gt[i] = split_train_dir
            train_images_arr.append(image)

            i += 1
    train_list.close()

    with open(test_list_path, 'r') as test_list:
        j = 0
        while True:
            label = test_list.readline().strip()
            if not label:
                break
            labels = cv2.imread(label)
            split_test_dir = int(label.split('/')[7])
            test_images_gt[j] = split_test_dir
            test_images[j] = labels
            j += 1
    test_list.close()
    # b = 0
    # for idx,i in enumerate(cls_list):
    #     cls_dir = os.path.join(path,i)
    #     img_list = os.listdir(cls_dir)
    #     for img in img_list:
    #         gt[b] = idx
    #         img_dir = os.path.join(cls_dir, img)
    #         temp_img = cv2.imread(img_dir)
    #         images[b] = temp_img
    #         b += 1
    return train_images_arr, train_images_gt, test_images, test_images_gt

def checknum(dirname):
    path = f"/home/aivs/바탕화면/dataset/Tiny_ImageNet/{dirname}/"

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
    #trans_imgs = torch.zeros((batch, 128, 128, 3), dtype=torch.float32)
    target = torch.zeros(batch, dtype=torch.int64)
    # rand_idx = np.random.randint(0, len(data), batch)
    # rand_idx = np.arange[0,len(data)]

    for idx,i in enumerate(data):
        img = cv2.imread(data[idx]) / 255 * 2.0 - 1.0
        if np.random.randint(2):
            img = cv2.flip(img, 1)
            #zero padding
            h, w, _ = img.shape
            padding_space = np.zeros((h+8, w+8, 3), dtype=img.dtype)
            padding_space[4:4+h, 4:4+w, :] = img
            img = padding_space
            #random Crop
            x_start = np.random.randint(0,9)
            y_start = np.random.randint(0,9)
            cropped_img = img[x_start:x_start+128, y_start:y_start+128, :]
            img = cropped_img


            #PCA_img = img.reshape(-1, 3)
        target[idx] = torch.tensor(gt[idx], dtype=torch.int32)
        imgs[idx] = torch.tensor(img, dtype=torch.float32)


    imgs = torch.permute(imgs, (0, 3, 1, 2))

    return imgs, target

# def data_loader(numofdata, dataval):
#     images = np.zeros((numofdata, 128, 128, 3), dtype=np.uint8)
#     gt = np.zeros(numofdata, dtype=np.uint8)
#
#     path = f"/home/aivs/바탕화면/dataset/Tiny_ImageNet/{dataval}/"
#
#     cls_list = os.listdir(path)
#
#     b = 0
#
#     for idx,i in enumerate(cls_list):
#         cls_dir = os.path.join(path,i)
#         img_list = os.listdir(cls_dir)
#         for img in img_list:
#             gt[b] = idx
#             img_dir = os.path.join(cls_dir, img)
#             temp_img = cv2.imread(img_dir)
#             images[b] = temp_img
#             b += 1
#     return images, gt
#
# def checknum(dirname):
#     path = f"/home/aivs/바탕화면/dataset/Tiny_ImageNet/{dirname}/"
#
#     cls_list = os.listdir(path)
#     q = 0
#     for i in cls_list:
#         img_dir = os.path.join(path,i)
#         img_list = os.listdir(img_dir)
#         for j in img_list:
#             q += 1
#     print(q)
#
# def mini_batch(data, gt, batch):
#     imgs = torch.zeros((batch, 128, 128, 3), dtype=torch.float32)
#     #trans_imgs = torch.zeros((batch, 128, 128, 3), dtype=torch.float32)
#     target = torch.zeros(batch, dtype=torch.int64)
#     rand_idx = np.random.randint(0, len(data), batch)
#
#     for i in range(len(rand_idx)):
#         img = data[rand_idx[i]] / 255 * 2.0 - 1.0
#         if np.random.randint(2):
#             img = cv2.flip(img, 1)
#             #zero padding
#             h, w, _ = img.shape
#             padding_img = np.zeros((h+8, w+8, 3), dtype=img.dtype)
#             padding_img[4:4+h, 4:4+w, :] = img
#             img = padding_img
#             #random Crop
#             x_start = np.random.randint(0,9)
#             y_start = np.random.randint(0,9)
#             cropped_img = img[x_start:x_start+128, y_start:y_start+128, :]
#             img = cropped_img
#
#             PCA_img = img.reshape(-1, 3)
#
#         target[i] = torch.tensor(gt[rand_idx[i]], dtype=torch.int32)
#         imgs[i] = torch.tensor(img, dtype=torch.float32)
#
#     imgs = torch.permute(imgs, (0, 3, 1, 2))
#
#     return imgs, target


# q,w = data_loader(51752,'val')
# a,b = mini_batch(q,w,16)
# print(a.shape)

