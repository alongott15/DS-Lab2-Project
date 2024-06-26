import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import argparse
import json
from dotmap import DotMap
import os
import time
import importlib as il

def create_dirs(dirs):
    try:
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
        return 0
    except Exception as e:
        print(e)
        return -1

def preprocess_input(x):
    return 2 * ((x / 255.0) - 0.5)

def eraser(input_img, p=0.5, sl=0.1, sh=0.3, r1=0.3, r2=1/0.1, vl=0, vh=255):
    input_img = preprocess_input(input_img)

    im_h, im_w, _ = input_img.shape

    pl = np.random.rand()
    if pl > p:
        return input_img
    
    while True:
        s = np.random.uniform(sl, sh) * im_h * im_h
        r = np.random.uniform(r1, r2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt( s * r))
        left = np.random.randint(0, w)
        top = np.random.randint(0, h)

        if left +  w <= im_w and top + h <= im_h:
            break

    c = np.random.uniform(vl, vh)
    input_img[top:(top + h), left:(left+w), :] = c
    return input_img

def get_generator(config, is_train=True):
    if config.data_loader.is_use_cutOut:
        preprocess = eraser(vl=0, vh=1)
    else:
        preprocess = preprocess_input
    
    if is_train:
        data_dir = config.data_loader.data_dir_train
    else:
        data_dir = config.data_loader.data_dir_val
    
    data_gen = ImageDataGenerator(
        preprocessing_function=preprocess,
        horizontal_flip=config.data_loader.horizontal_flip,
        fill_mode=config.data_loader.fill_mode,
        zoom_range=config.data_loader.zoom_range,
        width_shift_range=config.data_loader.width_shift_range,
        height_shift_range=config.data_loader.height_shift_range,
        rotation_range=config.data_loader.rotation_range
    )

    generator = data_gen.flow_from_directory(
        data_dir,
        target_size=(config.model.img_height, config.model.img_width),
        batch_size=config.data_loader.batch_size,
        class_mode=config.data_loader.class_mode
    )

    return generator

def get_test_generator(config, is_train=False, folder=''):
    if folder == '':
        if is_train:
            data_dir = config.data_loader.data_dir_train_test
        else:
            data_dir = config.data_loader.data_dir_valid_test
    else:
        data_dir = folder
        
    data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=False,
        vertical_flip=False
    )

    generator = data_gen.flow_from_directory(
        data_dir,
        target_size=(config.model.img_height, config.model.img_width),
        batch_size=config.data_loader.batch_size,
        class_mode='sparse',
        shuffle=False
    )

    return generator

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', 
        '--config',
        dest='config',
        metavar='C',
        default='None'
    )
    args = argparser.parse_args()
    return args

def get_config(json_file=''):
    with open(json_file, 'r') as f:
        config_dict = json.load(f)

    return DotMap(config_dict), config_dict

def process_config(json_file):
    config, _ = get_config(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp.name, "logs/")
    config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp.name, "checkpoints/")
    return config

def create(cls):
    module_name, class_name = cls.rsplit('.', 1)

    try:
        mod = il.import_module(module_name)
        cls_instance = GeneratorExit(mod, class_name)
    except Exception as e:
        print(e)
        return -1
    
    return cls_instance