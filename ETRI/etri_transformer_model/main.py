from load import * # type: ignore
from network import LocalGlobalMaskAwareTransformer # type: ignore
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

batch_size = 8
EPOCHS = 100
Iteration = 500
save_t = 10
restore = False
max_acc = 0
busan_seoul_02_000 = [33,160]
seoul_busan_02_000 = [160,33]
busan_seoul_02_001 = [32,160]
seoul_busan_02_001 = [160,32]
busan_seoul_01_000 = [33,88]
seoul_busan_01_000 = [88,33]
busan_seoul_01_001 = [33,88]
seoul_busan_01_001 = [88,33]
busan_seoul_01_002 = [33,86]
seoul_busan_01_002 = [86,33]
now_use = 'busan_seoul_01_002'
exp_name = [f'./layer4/{now_use}',f'{now_use}']
acc_name = f'{exp_name[1]}'
#데이터 불러오기
print(exp_name)
test_image_path = '/dj/seoul_txts/tiff_list_01_002.txt' #Seoul
test_gts_path = "/dj/seoul_txts/gt_list_01_002.txt" #Seoul

train_image_path = '/dj/busan_yrp_dataset/dataset_txt/start_dataset/tiff_list_002.txt' #Busan
train_gts_path = '/dj/busan_yrp_dataset/dataset_txt/start_dataset/gt_list_002.txt' #Busan
tr_num, ts_num = busan_seoul_01_002


train_images, train_gts, test_images, test_gts = sc_conv1d_dataset_load(train_image_path, train_gts_path, test_image_path, test_gts_path, tr_num, ts_num)
print('data load finish')

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(DEVICE)
model = LocalGlobalMaskAwareTransformer(p_drop=0.5).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.1)

def train(model, image_data, image_gt, batch_size, optimizer, DEVICE, Iteration):
    model.train()
    train_loss = 0
    for _ in range(Iteration):
        random_idx = np.random.randint(0,len(image_data), batch_size)
        batch_img, batch_target = mini_batch(image_data[random_idx], image_gt[random_idx]) # type: ignore
        bth_img, bth_target = batch_img.to(DEVICE), batch_target.squeeze().type(torch.float32).to(DEVICE)
        optimizer.zero_grad()
        output = model(bth_img)
        loss = criterion(output, bth_target)
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss
        loss.backward()
        optimizer.step()

    train_loss /= Iteration
    return train_loss

def evaluate(model, test_data, test_gt, DEVICE):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for bth in range(len(test_data)):
            bth_test = torch.from_numpy(np.array([test_data[bth]], dtype=np.float32))
            bth_test_target = torch.from_numpy(np.array([test_gt[bth]], dtype=np.float32))
            bth_test, bth_test_target = bth_test.to(DEVICE), bth_test_target.to(DEVICE)
            output = model(bth_test)
            loss = criterion(output, bth_test_target).item()
            prob = torch.sigmoid(output)
            test_loss += loss
            pred = (prob > 0.5).long()
            targ = bth_test_target.long()
            correct += (pred == targ).sum().item()

    test_loss /= len(test_data)
    test_accuracy = 100. * correct / len(test_data)
    return test_loss, test_accuracy


for epoch in range(1,EPOCHS+1):
    best_model = False
    if epoch % 50 == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    check_lr = optimizer.param_groups[0]['lr']
    train_loss = train(model, train_images, train_gts, batch_size, optimizer, DEVICE, Iteration)
    print("train done")
    test_loss, test_accuracy = evaluate(model, test_images, test_gts, DEVICE)
    if max_acc < test_accuracy:
        max_acc = test_accuracy
        print(max_acc)
        best_model = True
    with open(f'./model_accuracy/{acc_name}.txt' , 'a') as f:
        f.write('[{}]Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}\n'.format(epoch, test_loss, test_accuracy, check_lr))
    print('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
    print('[{}]Test Loss : {:.6f}, test Accuracy : {:.3f}%'.format(epoch, test_loss, test_accuracy))
    if epoch % save_t == 0:
        os.makedirs(f'{exp_name[0]}', exist_ok=True)
        torch.save(model, f'{exp_name[0]}/{epoch}_{exp_name[1]}.pt')
    if best_model:
        os.makedirs(f'{exp_name[0]}', exist_ok=True)
        torch.save(model, f'{exp_name[0]}/best_model({test_accuracy:.3f}).pt')
        print('best_model save')