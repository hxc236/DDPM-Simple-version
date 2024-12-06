import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from numpy.distutils.command.config import config
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader

from DDPM import DenoiseDiffusion
from config import TestConfig, Config
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


configs = TestConfig()

# 预处理
def preprocess_cifar10(dataset):
    """
    转换 CIFAR10 数据为 [-1, 1] 范围
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将数据缩放到 [-1, 1]
    ])
    return transforms.Compose([transform])(dataset)

# 保存图片
def save_images(images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, img in enumerate(images):
        save_image(img, os.path.join(save_dir, f"generated_{idx}.png"))


def test():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 CIFAR10 测试集
    transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    unet = torch.load(configs.save_dir)  # 替换为你的权重文件路径
    dm = DenoiseDiffusion(unet, n_steps=configs.n_steps, device=device)

    # FID 初始化
    # fid = FrechetInceptionDistance(feature=2048).to(device)

    # 生成图片并保存
    save_dir = "./output_images/generated_cifar10"
    generated_images = []
    real_images = []
    cur = 0
    for batch in tqdm(test_loader):
        cur += 1
        real_imgs, _ = batch
        real_imgs = real_imgs.to(device)
        real_images.append(real_imgs)

        # 从噪声生成图片
        noise = torch.randn(real_imgs.shape, device=device)
        with torch.no_grad():
            generated_imgs = dm.sample(noise)  # 调用 DDPM 推理方法生成 x0

        # 直接保存
        save_image(generated_imgs, os.path.join(save_dir, f"generated_{cur}.png"))

        # generated_images.append(generated_imgs)

        # 保存生成图片
    # save_images(torch.cat(generated_images, dim=0), save_dir)

    # 计算 FID
    # for real, fake in zip(real_images, generated_images):
    #     fid.update(real, real=True)
    #     fid.update(fake, real=False)

    # fid_score = fid.compute()
    # print(f"FID score: {fid_score}")


if __name__ == '__main__':
    test()