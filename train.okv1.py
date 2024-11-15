#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rockstar He
# Date: 2020-07-03
# Description:

import tensorflow.compat.v1 as tf
#//import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from liveness import create_model  # 确保这个模块是兼容tensorflow.keras的
from tensorflow.keras.optimizers import Adam
height = 32
width = 32
depth = 3

trainset = r'D:\wwwroot\github\moire_detection\checkpoint\test_recapture_bak\train'
valset = r'D:\wwwroot\github\moire_detection\checkpoint\test_recapture_bak\val'
ckp_path = r'models\ckp.h5'

def train():
    generator = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True
    )

    traindataloader = generator.flow_from_directory(
        trainset,
        batch_size=32,
        target_size=(height, width),
        color_mode='rgb',  # 确保color_mode设置为'rgb'，因为depth=3
        class_mode='categorical'  # 如果你的标签是多分类的，使用'categorical'
    )

    valdataloader = generator.flow_from_directory(
        valset,
        batch_size=32,
        target_size=(height, width),
        color_mode='rgb',
        class_mode='categorical'
    )

    train_ckp = ModelCheckpoint(ckp_path, monitor='val_accuracy')  # 使用'val_accuracy'代替'val_acc'

    optimizer = Adam(learning_rate=0.001)

    model = create_model()  # 确保这个函数返回一个兼容的tensorflow.keras模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print('=' * 40 + '开始训练' + '=' * 40)
    model.fit(
        traindataloader,
        epochs=100,
        verbose=1,
        callbacks=[train_ckp],
        validation_data=valdataloader,
        workers=2
    )



    model.save_weights(r'models\liveness3.1.h5')

if __name__ == "__main__":
    train()