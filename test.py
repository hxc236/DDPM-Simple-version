import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from numpy.distutils.command.config import config
from torch import nn
from torch.cuda import device

from DDPM import DenoiseDiffusion
from config import TestConfig


configs = TestConfig()

def show_sample(images, texts):
    _, figs= plt.subplots(1, len(images), figsize= (12, 12))
    for text, f, img in zip(texts, figs, images):
        f.imshow(img.view(configs.load_size, configs.load_size), cmap= 'gray')
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        f.text(0.5, 0, text, ha= 'center', va= 'bottom', fontsize= 12, color= 'white', backgroundcolor= 'black')
    plt.show()

def save_images(images, texts, save_dir):
    """
        保存生成的图片到本地
        :param images: 图片张量列表
        :param texts: 图片对应的步数标签
        :param save_dir: 保存路径
        """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果目录不存在，则创建

    for img, step in zip(images, texts):
        img = img.squeeze().cpu().numpy()  # 将图片张量转换为 NumPy 数组

        img_normalized1 = (img + 1) * 127.5  # 映射到 [0, 255]
        img_normalized1 = np.clip(img_normalized1, 0, 255).astype('uint8')  # 裁剪到 [0, 255] 并转换为 uint8 类型
        # 转换为 PIL 图像
        image = Image.fromarray(img_normalized1, mode='L')
        image.save(os.path.join(save_dir, f"sample_step_{step}_1.png"))



    print(f"Saved {len(images)} images to {save_dir}")


def test():
    save_dir = configs.save_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "./output_images"  # 保存图片的文件夹
    # 查看分配的内存
    # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")


    # xt 随机生成的正态分布的噪声
    xt, images, texts = torch.randn((1, 1, configs.load_size, configs.load_size), device=device), [], []
    unet = torch.load(f'{save_dir}')
    dm = DenoiseDiffusion(unet, 1000, device=device)

    # 查看分配的内存
    # print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e6} MB")

    for t in reversed(range(1000)):
        xt_1 = dm.p_sample(xt, torch.tensor([t]).to(device)).detach()       # .detach()确保它已经与计算图断开，否则PyTorch会延迟释放内存导致out of memory
        xt = xt_1
        del xt_1  # 删除不再需要的变量
        torch.cuda.empty_cache()    #
        if (t + 1) % 100 == 1:
            images.append(xt.view(1, configs.load_size, configs.load_size).to('cpu').detach())
            texts.append(t + 1)
        # 查看分配的内存
        # print("Memory reserved on t = {}: {} MB".format(t + 1, torch.cuda.memory_reserved() / 1e6))
    images_ = torch.stack(images, dim=0)
    show_sample(images_, texts)
    save_images(images_, texts, output_dir)  # 保存图片到本地

if __name__ == '__main__':
    test()