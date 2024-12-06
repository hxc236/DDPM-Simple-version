import os
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
import glob



IMG_EXTENSIONS = [
    '.jpg', '.png', '.jpeg', '.tiff'
]

def load_path(root):
    paths = []
    for file_type in IMG_EXTENSIONS:
        path = glob.glob(os.path.join(root, '*{}'.format(file_type)))
        if path:
            paths += path
    if len(paths) == 0:
        path = glob.glob(os.path.join(root, '*.npy'))
        if path:
            paths += path
    return paths



def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh/base)*base)
    w = int(round(ow/base)*base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)



def get_transform(conf, is2d=True, grayscale=False, method=Image.BICUBIC):
    transforms_list = []
    if grayscale:
        transforms_list.append(transforms.Grayscale(1))
    if 'resize' in conf.preprocess:
        osize = [conf.load_size, conf.load_size]
        transforms_list.append(transforms.Resize(osize, method))
    if 'centercrop' in conf.preprocess:
        transforms_list.append(
            transforms.CenterCrop(conf.crop_size)
        )
    if 'crop' in conf.preprocess:
        transforms_list.append(
            transforms.RandomCrop(conf.crop_size)
        )

    if conf.preprocess == 'none':
        transforms_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))
        )

    if conf.flip:
        transforms_list.append(
            transforms.RandomHorizontalFlip()
        )

    transforms_list.append(transforms.ToTensor())

    if is2d:
        transforms_list.append(
            transforms.Normalize((0.5, ), (0.5, ))
        )
    else:
        transforms_list.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )

    return transforms.Compose(transforms_list)


def transform_resize(conf, method=Image.BICUBIC):
    transforms_list = []
    osize = [conf.load_size, conf.load_size]
    transforms_list.append(transforms.Resize(osize, method))
    return transforms.Compose(transforms_list)


class MyDataset(data.Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.root = conf.dataroot
        self.dir = os.path.join(conf.dataroot, conf.type)
        self.paths =load_path(self.dir)
        self.len =len(self.paths)

        self.transform = get_transform(self.conf, grayscale=False)
        self.transform_resize = transform_resize(self.conf)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.basename(path)
        img = np.load(path)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = Image.fromarray(np.uint8(img)).convert('L')
        img = self.transform(img)
        img = self.transform_resize(img)

        return {'img': img, 'paths': path, 'name': name}


if __name__ == '__main__':
    from config import TrainConfig, ValidConfig
    config1 = TrainConfig()
    config2 = ValidConfig()
    data1 = MyDataset(config1)
    data2 = MyDataset(config2)
    sample = data1[1]['img']
    print("Min value:", sample.min().item())
    print("Max value:", sample.max().item())
    print(data1)
    print(data2)
