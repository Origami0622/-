import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # 小波变换的参数
        # nn.Parameter 是 PyTorch 中的一个类，用于将张量转换为模型的参数，使其能够在训练过程中通过反向传播自动更新。
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # self.scale：一个初始化为全1的张量，表示尺度参数。
        # self.translation：一个初始化为全0的张量，表示平移参数。
        # self.wavelet_weights：一个随机初始化的张量，表示小波权重。

        # 用于组合输出的线性权重
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # 批量规范化
        self.bn = nn.BatchNorm1d(out_features)
        # self.dropout = nn.Dropout(p=0.4)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)  # (batch_size, out_features, in_features)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # 实现不同的小波类型
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # 中心频率
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        elif self.wavelet_type == 'dog':
            # 实现高斯小波的导数
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # 在此处实现 Meyer Wavelet
            # Meyer 小波过渡边界的常数
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2,torch.ones_like(v),torch.where(v >= 1,torch.zeros_like(v),torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            # 使用辅助函数进行 Meyer 小波计算
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            # 对 sinc 函数进行窗口化以限制其支持
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # 应用 Hamming 窗口以限制对 sinc 函数的无限支持
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device)
            # Shannon 小波是 sinc 函数和窗口的乘积
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)

        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)

        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output + base_output

        # 应用批量规范化
        bn_out = self.bn(combined_output)
        # output = self.dropout(bn_out)
        return bn_out


class KAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, wavelet_type))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
