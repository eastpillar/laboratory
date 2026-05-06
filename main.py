from Load import mini_batch, data_loader
from network import VGG
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
import cv2

batch_size = 256
EPOCHS = 200
save_t = 10
restore = False
exp_name = 'model_accuracy'
#데이터 불러오기

train_list_path = '/home/aivs/바탕화면/adj_test/DenseNet/train_img_dir.txt' #Provide the TXT file path for your train image list.
test_list_path = '/home/aivs/바탕화면/adj_test/DenseNet/test_img_dir.txt' #Provide the TXT file path for your test image list.
train_img_arr, train_img_gt, test_img, test_img_gt = data_loader(train_list_path, test_list_path)

print('data load finish')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#학습 코드
# pretrained_model = '/home/aivs/바탕화면/adj_test/VGG/model_save/200_model.pt'

model = VGG().to(DEVICE)
# model.load_state_dict(torch.load(pretrained_model, map_location=DEVICE))

if restore:
    model.load_state_dict(torch.load(f'/home/aivs/바탕화면/adj_test/VGG/model_save'))

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)

def train(model, image_data, image_gt, batch_size, optimizer, DEVICE):
    model.train()
    image_gt = torch.tensor(image_gt, dtype=torch.int32)
    batch_split_img, batch_split_gt = [], []
    batch_split_img = np.array_split(image_data, math.ceil(len(image_data) / batch_size))
    batch_split_gt = np.array_split(image_gt, math.ceil(len(image_data) / batch_size))

    for batch_idx in tqdm(range(int(len(image_data) / batch_size))):
        batch_img, batch_target = mini_batch(batch_split_img[batch_idx], batch_split_gt[batch_idx], batch_size)
        bth_img, bth_target = batch_img.to(DEVICE), batch_target.to(DEVICE)
        optimizer.zero_grad()
        output = model(bth_img)

        loss = F.cross_entropy(output, bth_target)
        temp = 0
        loss.backward()
        optimizer.step()

def evaluate(model,test_data, test_gt, DEVICE):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for bth in tqdm(range(len(test_data))):
            bth_test = torch.from_numpy(np.array([test_data[bth]], dtype=np.float32))/255*2.0-1.0
            bth_test_target = torch.from_numpy(np.array([test_gt[bth]], dtype=np.int64))
            bth_test, bth_test_target = bth_test.permute(0, 3, 1, 2).to(DEVICE), bth_test_target.to(DEVICE)
            output, features = model(bth_test)
            feature = features.cpu().numpy()
            feature = (feature + 1) / 2 * 255
            for i in range(feature.shape[1]):
                each_feature = feature[:,i,:,:].squeeze()
                temp = 0
                cv2.imwrite(f"/home/aivs/바탕화면/hdd/dataset/VGG_features/feature_{i}.png", each_feature.astype(np.uint8))
            test_loss += F.cross_entropy(output, bth_test_target, reduction='sum').item()
            predict = output.max(1, keepdim=True)[1]
            correct += predict.eq(bth_test_target.view_as(predict)).sum().item()

    test_loss /= len(test_data)
    test_accuracy = 100. * correct / len(test_data)
    return test_loss, test_accuracy

for epoch in range(1,EPOCHS+1):
    # train(model, train_img_arr, train_img_gt, batch_size, optimizer, DEVICE)
    print("train done")
    test_loss, test_accuracy = evaluate(model, test_img, test_img_gt, DEVICE)
    print("test done") #255 * 2.0 - 1.0
        # if np.random.randint(2):
        #     img = cv2.flip(img, 1)
        #     #zero padding
        #     h, w, _ = img.shape
        #     padding_space = np.zeros((h+8, w+8, 3), dtype=img.dtype)
        #     padding_space[4:4+h, 4:4+w, :] = img
        #     img = padding_space
        #     #random Crop
        #     x_start = np.random.randint(0,9)
        #     y_start = np.random.randint(0,9)
        #     cropped_img = img[x_start:x_start+128, y_start:y_start+128, :]
        #     img = cropped_img


    with open(f'./model_accuracy/{exp_name}.txt' , 'a') as f:
        f.write(f'Epoch: {epoch} >> Acc: {test_accuracy}\n')
    print('[{}] Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(epoch, test_loss, test_accuracy))
    if epoch % save_t == 0:
        torch.save(model.state_dict(), f'./model_save/{epoch}_model.pt')
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
