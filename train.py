import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
from torch import nn
import matplotlib.pyplot as plt

import config
from DDPM import DenoiseDiffusion
from DDPM_parts import UNet
from config import TrainConfig, ValidConfig
from dataset import MyDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    config1 = TrainConfig()
    config2 = ValidConfig()
    train_dataset = MyDataset(config1)
    valid_dataset = MyDataset(config2)
    train_loader = DataLoader(train_dataset, batch_size=config1.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config2.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    unet = UNet(config1.in_channel, config1.n_channels, [1, 2, 2, 4], [False, False, False, True]).to(device)

    dm = DenoiseDiffusion(unet, config1.n_steps, device=device)
    # opt_dm = torch.optim.Adam(unet.parameters(), lr=config1.lr, weight_decay=1e-3)
    opt_dm = torch.optim.Adam(unet.parameters(), lr=config1.lr)

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - config1.epochs / 2) / float(config1.epochs / 2 + 1)
        return lr_l
    # 学习率：线性衰减 轮数超过一半开始衰减
    scheduler = LambdaLR(opt_dm, lr_lambda=lambda_rule)
    # # 使用余弦退火学习率调度器
    # scheduler = CosineAnnealingLR(opt_dm, T_max=config1.epochs, eta_min=1e-6)

    best_score = config1.best_score
    early_stop_threshold = config1.early_stop_threshold
    early_stop_time = 0
    train_losses, valid_losses = [], []

    print(train_loader)

    for epoch in range(config1.epochs):
        loss_record = []
        iteration = 0

        print("Training Epoch {} / {}, Current lr = {}".format(epoch, config1.epochs, scheduler.get_last_lr()))

        for step, batch in enumerate(train_loader):
            # print(batch)
            pic = batch['img'].to(device)

            # debug code
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
            for step, batch in enumerate(valid_loader):
                pic = batch['img'].to(device)
                loss = dm.loss(pic)
                loss_record.append(loss.item())
        mean_loss = torch.tensor(loss_record).mean()
        valid_losses.append(mean_loss.item())
        # early stopping
        if mean_loss < best_score:
            early_stop_time = 0
            best_score = mean_loss
            torch.save(unet, f'{config1.save_dir}')
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