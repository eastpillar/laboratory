import cv2 # type: ignore
import os
import numpy as np
import torch

def similarity_map(image):
    sim_image = np.dot(image.T, image)
    temp = 0
    return sim_image



def sc_conv1d_dataset_load(train_image_path, train_gt_txt, test_image_path, test_gt_txt, tr_num, ts_num):
    tr_images = np.zeros((tr_num, 6, 300), dtype=np.float32)
    tr_gts = np.zeros((tr_num), dtype=np.uint8)
    ts_images = np.zeros((ts_num, 6, 300), dtype=np.float32) # [수정 1]
    ts_gts = np.zeros((ts_num), dtype=np.uint8)
    
    mask_value = -50.0

    with open(train_image_path, 'r') as tr_imgs:
        i = 0
        while True:
            img = tr_imgs.readline().strip()
            if not img:
                break
            
            image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            yaw_img, roll_img, pitch_img = image[0,:], image[1,:], image[2,:]

            yaw_img_resized = cv2.resize(yaw_img, (1, 300), interpolation=cv2.INTER_NEAREST)
            roll_img_resized = cv2.resize(roll_img, (1, 300), interpolation=cv2.INTER_NEAREST)
            pitch_img_resized = cv2.resize(pitch_img, (1, 300), interpolation=cv2.INTER_NEAREST)

            yaw_is_zero = (yaw_img_resized == 0.0)
            roll_is_zero = (roll_img_resized == 0.0)
            pitch_is_zero = (pitch_img_resized == 0.0)
            
            zero_mask = (yaw_is_zero & roll_is_zero & pitch_is_zero).squeeze()

            yaw_img_rad = yaw_img_resized * np.pi / 180.0
            roll_img_rad = roll_img_resized * np.pi / 180.0
            pitch_img_rad = pitch_img_resized * np.pi / 180.0

            yaw_sincos = np.stack((np.sin(yaw_img_rad), np.cos(yaw_img_rad)), axis=0).squeeze() # (2, 300)
            roll_sincos = np.stack((np.sin(roll_img_rad), np.cos(roll_img_rad)), axis=0).squeeze() # (2, 300)
            pitch_sincos = np.stack((np.sin(pitch_img_rad), np.cos(pitch_img_rad)), axis=0).squeeze() # (2, 300)

            stack_img = np.concatenate((yaw_sincos, roll_sincos, pitch_sincos), axis=0) # (6, 300)

            stack_img[:, zero_mask] = mask_value
            tr_images[i] = stack_img
            i += 1


    with open(train_gt_txt, 'r') as tr_g:
        j = 0
        while True:
            gt = tr_g.readline().strip()
            if not gt:
                break
            tr_gts[j] = int(gt)
            j += 1

    with open(test_image_path, 'r') as tr_imgs:
        i = 0
        while True:
            img = tr_imgs.readline().strip()
            if not img:
                break
            
            image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            yaw_img, roll_img, pitch_img = image[0,:], image[1,:], image[2,:]

            yaw_img_resized = cv2.resize(yaw_img, (1, 300), interpolation=cv2.INTER_NEAREST)
            roll_img_resized = cv2.resize(roll_img, (1, 300), interpolation=cv2.INTER_NEAREST)
            pitch_img_resized = cv2.resize(pitch_img, (1, 300), interpolation=cv2.INTER_NEAREST)

            yaw_is_zero = (yaw_img_resized == 0.0)
            roll_is_zero = (roll_img_resized == 0.0)
            pitch_is_zero = (pitch_img_resized == 0.0)
            
            zero_mask = (yaw_is_zero & roll_is_zero & pitch_is_zero).squeeze() 

            yaw_img_rad = yaw_img_resized * np.pi / 180.0
            roll_img_rad = roll_img_resized * np.pi / 180.0
            pitch_img_rad = pitch_img_resized * np.pi / 180.0

            yaw_sincos = np.stack((np.sin(yaw_img_rad), np.cos(yaw_img_rad)), axis=0).squeeze()
            roll_sincos = np.stack((np.sin(roll_img_rad), np.cos(roll_img_rad)), axis=0).squeeze()
            pitch_sincos = np.stack((np.sin(pitch_img_rad), np.cos(pitch_img_rad)), axis=0).squeeze()

            stack_img = np.concatenate((yaw_sincos, roll_sincos, pitch_sincos), axis=0)

            stack_img[:, zero_mask] = mask_value
            
            ts_images[i] = stack_img
            i += 1


    with open(test_gt_txt, 'r') as tr_g:
        j = 0
        while True:
            gt = tr_g.readline().strip()
            if not gt:
                break
            ts_gts[j] = int(gt)
            j += 1

    return tr_images, tr_gts, ts_images, ts_gts


def mini_batch(images, gts, jitter_prob=0.5, jitter_std=0.05):

    to_tensor_img = torch.tensor(np.array(images, dtype=np.float32))
    to_tensor_gt  = torch.tensor(np.array(gts, dtype=np.int64))

    if np.random.randint(2):
        noise = torch.randn_like(to_tensor_img) * jitter_std
        mask = (to_tensor_img != -50.0)
        to_tensor_img = torch.where(mask, to_tensor_img + noise, to_tensor_img)

    if np.random.randint(2):
        s = np.random.randint(-20, -20 + 40)
        if s != 0:
            to_tensor_img = torch.roll(to_tensor_img, shifts=s, dims=2)

            if s > 0:
                to_tensor_img[:, :, :s] = -50
            elif s < 0:
                to_tensor_img[:, :, s:] = -50

    if np.random.randint(2):
        
        mask_len = np.random.randint(1, 30 + 1)

        if 300 > mask_len:
            mask_start = np.random.randint(0, 300 - mask_len + 1)
            
            to_tensor_img[:, :, mask_start : mask_start + mask_len] = -50

    return to_tensor_img, to_tensor_gt
