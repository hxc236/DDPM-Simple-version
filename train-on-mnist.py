import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
from torch import nn
import matplotlib.pyplot as plt

from DDPM import DenoiseDiffusion
from DDPM_parts import UNet
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import Config

def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = Config()

    config.in_channel = 1
    config.n_channels = 64
    config.batch_size = 256
    config.early_stop_threshold = 40
    config.epochs = 200
    config.save_dir = 'ckpt/unet_mnist.pt'

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))  # 标准化到 [-1, 1]
    ])

    # 加载完整的训练数据集
    full_trainset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 数据集划分比例
    train_size = int(0.8 * len(full_trainset))  # 80% 的数据
    valid_size = int(0.2 * len(full_trainset))  # 剩余 20% 的数据
    # others_size = len(full_trainset) - train_size - valid_size

    # 随机划分数据集
    train_dataset, valid_dataset = random_split(full_trainset, [train_size, valid_size])

    # 打印划分结果
    print(f"训练集大小: {len(train_dataset)}")  # 输出: 40000
    print(f"验证集大小: {len(valid_dataset)}")  # 输出: 10000

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    unet = UNet(config.in_channel, config.n_channels, [1, 2, 2], [False, False, False]).to(device)

    dm = DenoiseDiffusion(unet, config.n_steps, device=device)
    # opt_dm = torch.optim.Adam(unet.parameters(), lr=config1.lr, weight_decay=1e-3)
    opt_dm = torch.optim.Adam(unet.parameters(), lr=config.lr)

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - config.epochs / 2) / float(config.epochs / 2 + 1)
        return lr_l
    # 学习率：线性衰减 轮数超过一半开始衰减
    scheduler = LambdaLR(opt_dm, lr_lambda=lambda_rule)


    best_score = config.best_score
    early_stop_threshold = config.early_stop_threshold
    early_stop_time = 0
    train_losses, valid_losses = [], []

    print(train_loader)

    for epoch in range(config.epochs):
        loss_record = []
        iteration = 0

        print("Training Epoch {} / {}, Current lr = {}".format(epoch, config.epochs, scheduler.get_last_lr()))

        for step, (pic, label) in enumerate(train_loader):
            # print(batch)
            pic = pic.to(device)

            # debug code
            # print(pic.shape)
            # print("Input mean: {}, std: {}".format(pic.mean(), pic.std()))
            # debug end

            opt_dm.zero_grad()
            loss = dm.loss(pic)

            # debug code
            # print("Loss value: {}".format(loss.item()))
            # debug end

            loss_record.append(loss.item())
            loss.backward()
            opt_dm.step()

            iteration += 1
            if iteration % 10 == 0:
                print("iteration {} : mse loss = {} ".format(iteration, loss.item()))

        # 更新学习率
        scheduler.step()
        print("train mean loss: {}".format(torch.tensor(loss_record).mean()))
        train_losses.append(torch.tensor(loss_record).mean().item())

        loss_record = []
        with torch.no_grad():
            for step, (pic, label) in enumerate(valid_loader):
                pic = pic.to(device)
                loss = dm.loss(pic)
                loss_record.append(loss.item())
        mean_loss = torch.tensor(loss_record).mean()
        valid_losses.append(mean_loss.item())
        # early stopping
        if mean_loss < best_score:
            early_stop_time = 0
            best_score = mean_loss
            torch.save(unet, f'{config.save_dir}')
        else:
            early_stop_time = early_stop_time + 1
        if early_stop_time > early_stop_threshold:
            break
        # output
        print("early_stop_time/early_stop_threshold: {} / {}, valid mean loss: {}".format(early_stop_time, early_stop_threshold, mean_loss))

    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()