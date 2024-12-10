import torch
import torch.nn as nn
from .unet_parts import DownSample, Doubleconv, UpConv

class Unet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.Conv1 = Doubleconv(input_dim, 64, 64)
        self.down1 = DownSample()

        self.Conv2 = Doubleconv(64, 128, 128)
        self.down2 = DownSample()

        self.Conv3 = Doubleconv(128, 256, 256)
        self.down3 = DownSample()

        self.Conv4 = Doubleconv(256, 512, 512)
        self.down4 = DownSample()

        self.Conv5 = Doubleconv(512, 1024, 1024)

        self.upConv1 = UpConv(1024, 512)
        self.Conv6 = Doubleconv(1024, 512, 512)

        self.upConv2 = UpConv(512, 256)
        self.Conv7 = Doubleconv(512, 256, 256)

        self.upConv3 = UpConv(256, 128)
        self.Conv8 = Doubleconv(256, 128, 128)

        self.upConv4 = UpConv(128, 64)
        self.Conv9 = Doubleconv(128, 64, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        clc_x = self.Conv1(x)
        x_cat1 = clc_x
        clc_x = self.down1(clc_x)

        clc_x = self.Conv2(clc_x)
        x_cat2 = clc_x
        clc_x = self.down2(clc_x)

        clc_x = self.Conv3(clc_x)
        x_cat3 = clc_x
        clc_x = self.down3(clc_x)

        clc_x = self.Conv4(clc_x)
        x_cat4 = clc_x
        clc_x = self.down4(clc_x)

        clc_x = self.Conv5(clc_x)

        clc_x = torch.cat((x_cat4, self.upConv1(clc_x)), dim=1)
        clc_x = self.Conv6(clc_x)

        clc_x = torch.cat((x_cat3, self.upConv2(clc_x)), dim=1)
        clc_x = self.Conv7(clc_x)

        clc_x = torch.cat((x_cat2, self.upConv3(clc_x)), dim=1)
        clc_x = self.Conv8(clc_x)

        clc_x = torch.cat((x_cat1, self.upConv4(clc_x)), dim=1)
        clc_x = self.Conv9(clc_x)

        out_put = self.out_conv(clc_x)

        return out_put


if __name__ == "__main__":
    arr1 = torch.randn(5, 3, 512, 512).to(device=torch.device("cuda"))
    models = Unet(3).cuda()
    arr2 = models(arr1)

    print(arr2.shape)