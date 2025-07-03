#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is to implement deep fingerprinting model for website fingerprinting attacks
# ACM Reference Formant
# Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. 2018.
# Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.
# In 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS ’18),
# October 15–19, 2018, Toronto, ON, Canada. ACM, New York, NY, USA, 16 pages.
# https://doi.org/10.1145/3243734.3243768

# encoding=utf8

from keras import backend as K
#from Model_NoDef import DFNet
from Model_NoDef import DFNet
import random

import pickle
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint
#from keras.layers import ELU, PReLU, LeakyReLU
#from keras.optimizers import Adamax
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import numpy as np
import os
from collections import defaultdict
import gc

random.seed(0)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use GPU 

# Use only CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
feature = 'tiktok_TS3_258'
#feature = 'TAMoverlap_2_doubled_concat_size_2D_2000-60'
# path = '3'
NB_EPOCH = 30   # Number of training epoch

# data path
X_path = '/home/jiwoo0914/DF/DF/dataModel_4/Trace/part1/py/TS_5_X.pkl'
y_path = '/home/jiwoo0914/DF/DF/dataModel_4/Trace/part1/py/TS_5_Y.pkl'

def LoadData(X_path, y_path):

        # X_train , X_valid, X_test
        with open(X_path, 'rb') as f:
                X = pickle.load(f)
                                                                                                             
        # y_train, y_valid, y_test
        with open(y_path, 'rb') as f:
                y = pickle.load(f)

    # 지정된 클래스 수에 따라 클래스 선택
        selected_classes = range(258)

    # 각 클래스별 데이터 인덱스를 저장할 딕셔너리
        class_indices = defaultdict(list)
        for i, label in enumerate(y):
                if label in selected_classes:
                        class_indices[label].append(i)
    
    # 선택된 인덱스 초기화
        selected_indices = []
    # # 지정된 클래스에서 최대 1000개의 데이터만 선택
        # for label in selected_classes:
        #       if label in class_indices and len(class_indices[label]) > 1000:
        #               selected_indices.extend(class_indices[label][:1000])  # 처음 1000개 선택
        #       elif label in class_indices:
        #               selected_indices.extend(class_indices[label])

        # 지정된 클래스에서 모두 선택
        for label in selected_classes:
                selected_indices.extend(class_indices[label])

    # 데이터와 레이블을 선택된 인덱스로 필터링
        X = X[selected_indices]
        y = y[selected_indices]

        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

        X_valid, X_test, y_valid, y_test=train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        print("train: ", X_train.shape, end='\t')
        print(y_train.shape)
        print("valid: ", X_valid.shape, end='\t')
        print(y_valid.shape)
        print("test: ", X_test.shape, end='\t')
        print(y_test.shape)

        gc.collect()
        return X_train, y_train, X_valid, y_valid, X_test, y_test

description = "Training and evaluating DF model for closed-world scenario on TS dataset"

print(description)
# Training the DF model
print("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 128 # Batch size default 128
VERBOSE = 2 # Output display mode
LENGTH = 5000 # Packet sequence length
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

NB_CLASSES = 258 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)
#INPUT_SHAPE = (2,LENGTH,1)
#INPUT_SHAPE = (LENGTH,2) # add channels


# Data: shuffled and split between train and test sets
print("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test = LoadData(X_path, y_path)
# Please refer to the dataset format in readme
###########K.set_image_dim_ordering("tf") # tf is tensorflow
K.set_image_data_format('channels_first')

# Convert data as float32 type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')

# # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to categorical classes matrices
y_train = to_categorical(y_train, NB_CLASSES)
y_valid = to_categorical(y_valid, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

# Building and training model
print("Building and training DF model")

model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
# model = ConvNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
        metrics=["accuracy"])
print("Model compiled")

# Start training
# history = model.fit(X_train, y_train,
#               batch_size=BATCH_SIZE, epochs=NB_EPOCH,
#               verbose=VERBOSE, validation_data=(X_valid, y_valid))

# filepath = 'trainedmodel/CW_DF_'+feature+path+'_'+str(NB_EPOCH)+'.h5'
filepath = '/home/jiwoo0914/DF/DF/DF_model/60keywords_1000ins_'+feature+'_epoch'+str(NB_EPOCH)+'.h5'


checkpoint = ModelCheckpoint(filepath = filepath, 
                                                 monitor='val_accuracy',
                             verbose=2, 
                             save_best_only=True,
                             mode='max')
callbacks = [checkpoint]

# Start training
history = model.fit(X_train, y_train,
                batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                validation_data=(X_valid, y_valid),
        callbacks=callbacks)
gc.collect()
# Save & reload model
model.save(filepath)
del model
gc.collect()
model = load_model(filepath)

# Start evaluating model with testing data
score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
score_train = model.evaluate(X_train, y_train, verbose=VERBOSE)

print("\nEpoch: ", NB_EPOCH, ", Batch: ", BATCH_SIZE, ", feature: ", feature)

print(model.metrics_names)

print("\n=> Train score:", score_train[0])
print("=> Train accuracy:", score_train[1])

print("\n=> Test score:", score_test[0])
print("=> Test accuracy:", score_test[1])

# print("Testing accuracy:", score_test[1])

with open(f"/home/ifetayo/DF-original/results/{feature}.txt", 'w') as file:
    # Using f-string for better readability and direct inclusion of variables
    file.write(f"\nEpoch: {NB_EPOCH}, Batch: {BATCH_SIZE}, feature: {feature}")
    
    # Assuming model.metrics_names is a list of strings
    file.write("\n" + ", ".join(model.metrics_names))
    
    # Using f-strings for the rest of the writes
    file.write(f"\n=> Train score: {score_train[0]}")
    file.write(f"=> Train accuracy: {score_train[1]}")
    file.write(f"\n=> Test score: {score_test[0]}")
    file.write(f"=> Test accuracy: {score_test[1]}")