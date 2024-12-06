
class Config(object):
    batch_size = 10
    lr = 0.0001
    n_steps = 1000
    in_channel = 1
    n_channels = 64

    save_dir = 'ckpt/unet_cifar10.pt'

    # train
    best_score = 1e10
    epochs = 300
    early_stop_threshold = 80



class TrainConfig(Config):
    # preprocess
    load_size = 128
    # crop_size = 240
    flip = False
    serial_batch = False
    preprocess = 'resize'
    dataroot = r'D:/PreparedData/BrainT1T2FT/npyFTTT'

    type = 'T1'

    save_dir = 'ckpt/unet.pt'

class ValidConfig(Config):
    # preprocess
    load_size = 128
    # crop_size = 240
    flip = False
    serial_batch = False
    preprocess = 'resize'
    dataroot = r'D:/PreparedData/BrainT1T2FT/npyFTTTest'
    type = 'T1'

class TestConfig(Config):
    # preprocess
    load_size = 128
    # crop_size = 240
    flip = False
    serial_batch = False
    preprocess = 'resize'
    dataroot = r'D:/PreparedData/BrainT1T2FT/npyFTTTest'
    type = 'T1'