#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rockstar He
# Date: 2020-07-03
# Description:

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from liveness import create_model
from tensorflow.keras.optimizers import Adam

import os
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
        color_mode='rgb',
        class_mode='categorical'
    )

    valdataloader = generator.flow_from_directory(
        valset,
        batch_size=32,
        target_size=(height, width),
        color_mode='rgb',
        class_mode='categorical'
    )

    CHECKPOINT = 'models'
    filepath = os.path.join(CHECKPOINT, "weights-{epoch:02d}-{val_accuracy:.4f}.hdf5")

    train_ckp = ModelCheckpoint(
        filepath,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=False,  # 保存所有epochs的模型，而不只是最好的
        mode='max',
        period=5  # 每5个epochs保存一次模型
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        min_delta=0.001  # 设置一个最小变化值，以减少因小的波动而停止训练的情况
    )

    optimizer = Adam(learning_rate=0.001)
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print('=' * 40 + '开始训练' + '=' * 40)
    model.fit(
        traindataloader,
        epochs=100,
        verbose=1,
        callbacks=[train_ckp, early_stopping],  # 使用两个回调
        validation_data=valdataloader,
        workers=1# linux 才可以多线程 2 windows 改为 1
    )

if __name__ == "__main__":
    train()