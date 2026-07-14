import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

class ConvNet(nn.Module):
    def __init__(self, num_classes=11):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            # 卷积块1
            nn.Conv2d(1, 256, kernel_size=3, padding=1),  # 输入通道3，输出256
            nn.BatchNorm2d(256),  # 批量归一化
            nn.ReLU(inplace=True),  # 原地激活节省内存
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),  # 随机丢弃20%神经元

            # 卷积块2
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # 卷积块3
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平多维特征
            nn.Linear(64 * 7 * 7, 256),  # 全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # 输出层
        )

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = self.global_pool(x)
        x = self.classifier(x)  # 分类决策
        return x
