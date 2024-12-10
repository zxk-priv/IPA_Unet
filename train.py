import numpy as np
import torch

from model import Unet
from read_data import unet_dataset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import glob
from PIL import Image

# 训练权重存储位置
best_weight_path = './model_weight'
os.makedirs(best_weight_path, exist_ok=True)

# 测试结果存储位置
output_result = './results'
os.makedirs(output_result, exist_ok=True)

# 判断是否有可用的gpu设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [
        transforms.ToTensor()   # 慎用  会将读取的图像从[HWC]->[CHW]  而且也会归一化经过此处理的数据
    ]
)

train_dataset = unet_dataset('./data/train', transform=transform)   # 加载训练数据集
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=32)   # 分配batchsize

'''
函数名字：evaluate
参数：weight_path加载权重的路径  如：./model_weight/best_model.pth
功能：预测评估./data/test/*.png的图片，然后将预测结果的图片保存到./results下
'''
def evaluate(weight_path):
    models = Unet(1)
    models.to(device=device)
    models.load_state_dict(torch.load(weight_path, map_location=device))

    models.eval()

    img_path_list = glob.glob('./data/test/*.png')

    with torch.no_grad():
        for img_path in img_path_list:
            img = Image.open(img_path)
            gray_img = img.convert('L')
            img_arr = np.array(gray_img)
            img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 1)
            img_tensor = transform(img_arr)
            img_tensor = img_tensor.reshape(1, 1, img_arr.shape[0], img_arr.shape[1])
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)

            pred_img = models(img_tensor)

            pred_img = pred_img.cpu().numpy()

            pred_img = pred_img.reshape(img_arr.shape[0], img_arr.shape[1])

            pred_img[pred_img > 0.5] = 255

            pred_img[pred_img <= 0.5] = 0

            pred_img = Image.fromarray(pred_img.astype(np.uint8))

            pred_img.save(os.path.join(output_result, os.path.basename(img_path)))



'''
函数名字：train
参数：epochs 训练的迭代次数
    lr   学习率
功能：训练模型，然后将训练后的权重保存到./model_weight/下
'''
def train(epochs=100, lr=1e-3):
    models = Unet(1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(models.parameters(), lr=lr)

    models.to(device=device)
    models.train()

    loss_list = []
    best_loss = float('inf')
    for epoch in range(epochs):
        running_loss = 0.0
        for img, label in train_dataloader:
            img = img.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()

            pred_img = models(img)

            loss = criterion(pred_img, label)
            running_loss += loss.item()

            # 保存损失最小的网络
            if loss < best_loss:
                best_loss = loss
                torch.save(models.state_dict(), os.path.join(best_weight_path, 'best_model.pth'))

            loss.backward()

            optimizer.step()

        print(f"epoch={epoch}, loss={running_loss}\n")
        loss_list.append(running_loss)

    # 绘制loss曲线
    plt.plot(loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(best_weight_path, 'loss.png'))

if __name__ == "__main__":
    # 进行训练
    train(epochs=30)
    # 进行预测
    evaluate(os.path.join(best_weight_path, 'best_model.pth'))