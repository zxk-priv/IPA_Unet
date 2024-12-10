import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np

class unet_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, "image/*.png"))  # 返回所有的图片路径

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')

        read_img = Image.open(image_path)
        gray_img = np.array(read_img.convert('L'))
        gray_img = gray_img.reshape(gray_img.shape[0], gray_img.shape[1], 1)

        read_label = Image.open(label_path)
        gray_label = np.array(read_label.convert('L'))
        gray_label = gray_label.reshape(1, gray_label.shape[0], gray_label.shape[1])
        gray_label = torch.tensor(gray_label)

        if gray_label.max() > 1:
            gray_label = gray_label // 255

        if self.transform:
            gray_img = self.transform(gray_img)

        return gray_img, gray_label

    def __len__(self):
        return len(self.imgs_path)

if __name__ == "__main__":
    unet_data = unet_dataset('./data/train')
    for img, label in unet_data:
        print(img.shape)
        print(label.shape)




