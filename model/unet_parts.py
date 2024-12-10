import torch
import torch.nn as nn

# 前向卷积层
class Doubleconv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x:torch.Tensor):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.relu2(conv2)

        return conv2

# 下采样
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x: torch.Tensor):
        max_pool = self.max_pool(x)

        return max_pool

# 上采样
class UpConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        up_conv1 = self.up_conv(x)

        return up_conv1


if __name__ == "__main__":
    arr1 = torch.randn(20, 3, 16, 16)
    down = UpConv(3, 64)
    arr2 = down(arr1)
    print(arr2.shape)
    

