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

def test():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载  测试集
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    unet = torch.load(configs.save_dir)  # 替换为你的权重文件路径
    dm = DenoiseDiffusion(unet, n_steps=configs.n_steps, device=device)

    # FID 初始化
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # 生成图片并保存
    save_dir = "./output_images/generated_mnist"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

        generated_images.append(generated_imgs)

        # 保存生成图片
    # save_images(torch.cat(generated_images, dim=0), save_dir)

    # 计算 FID
    for real, fake in zip(real_images, generated_images):
        fid.update(real, real=True)
        fid.update(fake, real=False)

    fid_score = fid.compute()
    print(f"FID score: {fid_score}")


def test_one():

    configs.load_size = 32
    configs.in_channels = 1
    configs.save_dir = './ckpt/unet_mnist.pt'

    def show_sample(images, texts):
        _, figs = plt.subplots(1, len(images), figsize=(12, 12))
        for text, f, img in zip(texts, figs, images):
            f.imshow(img.view(configs.load_size, configs.load_size), cmap='gray')
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
            f.text(0.5, 0, text, ha='center', va='bottom', fontsize=12, color='white', backgroundcolor='black')
        plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    unet = torch.load(configs.save_dir)  # 替换为你的权重文件路径
    dm = DenoiseDiffusion(unet, n_steps=configs.n_steps, device=device)

    # xt 随机生成的正态分布的噪声
    xt, images, texts = torch.randn((1, 1, configs.load_size, configs.load_size), device=device), [], []
    dm = DenoiseDiffusion(unet, 1000, device=device)

    for t in reversed(range(1000)):
        xt_1 = dm.p_sample(xt, torch.tensor([t]).to(
            device)).detach()  # .detach()确保它已经与计算图断开，否则PyTorch会延迟释放内存导致out of memory
        xt = xt_1
        del xt_1  # 删除不再需要的变量
        torch.cuda.empty_cache()  #
        if (t + 1) % 100 == 1:
            images.append(xt.view(1, configs.load_size, configs.load_size).to('cpu').detach())
            texts.append(t + 1)

    images_ = torch.stack(images, dim=0)
    show_sample(images_, texts)

if __name__ == '__main__':
    test()
    # test_one()