from Load import mini_batch, data_loader
from CBAM_network import DenseNet121
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

batch_size = 128
EPOCHS = 150
save_t = 10
restore = False
exp_name = 'dense_CBAM_121_accuracy'
#데이터 불러오기
train_list_path = '/dataset/Tiny_ImageNet/train_img_dir.txt'
test_list_path = '/dataset/Tiny_ImageNet/test_img_dir.txt'
train_img_arr, train_img_gt, test_img, test_img_gt = data_loader(train_list_path, test_list_path)

# train_data, train_gt = data_loader(207005, 'train')
# test_data, test_gt = data_loader(51752, 'val')

print('finish load data')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(DEVICE)


model = DenseNet121(growth_rate=32, comp_factor=0.5).to(DEVICE)
if restore:
    model.load_state_dict(torch.load(f'{restore}_model.pth'))

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
#scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1)

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
        output = model(bth_img)
        loss = F.cross_entropy(output, bth_target)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= len(image_data)
    return train_loss


def evaluate(model,test_data, test_gt, DEVICE):
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
            #print(predict)
            correct += predict.eq(bth_test_target.view_as(predict)).sum().item()

    test_loss /= len(test_data)
    test_accuracy = 100. * correct / len(test_data)
    return test_loss, test_accuracy

for epoch in range(1,EPOCHS+1):
    if epoch == int(EPOCHS*0.5) or epoch == int(EPOCHS*0.75):
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    check_lr = optimizer.param_groups[0]['lr']
    train_loss = train(model, train_img_arr, train_img_gt, batch_size, optimizer, DEVICE)
    print("train done")
    test_loss, test_accuracy = evaluate(model, test_img, test_img_gt, DEVICE)
    #scheduler.step(test_loss)
    with open(f'./model_accuracy/{exp_name}.txt' , 'a') as f:
        f.write('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}\n'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
    print("test done")
    print('[{}]Train Loss : {:.6f}, Test Loss : {:.6f}, test Accuracy : {:.3f}%, LR : {:.6f}'.format(epoch, train_loss, test_loss, test_accuracy, check_lr))
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