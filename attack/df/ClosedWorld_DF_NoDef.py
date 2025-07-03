#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Deep Fingerprinting model for website fingerprinting attacks

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from keras import backend as K
from XAI_WF.attack.df.plot_result import load_pkl
# from Model_Leaky import LeakyDFNet
from Model_Focal import DFNet
import random
from tensorflow.keras.optimizers import Adamax
import numpy as np
import matplotlib.pyplot as plt
import os
from XAI_WF.attack.df.plot_result import plot_history, plot_corr
from sklearn.metrics import classification_report
import pandas as pd
from data_loader import load_raw

random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

description = "Training and evaluating DF model for closed-world scenario on non-defended dataset"
print(description)

# 설정값
feature = "size"
NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 1
LENGTH = 5000
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
INPUT_SHAPE = (LENGTH, 1)

# 제외할 클래스 설정
# zero_classes = [9, 15, 18, 28, 34, 36, 37, 39, 48, 51, 53, 56, 57, 64, 67]
entire_classes = set(range(75))  # 리스트를 집합으로 변경
zero_classes = set([])  # 빈 집합으로 설정
excluded_classes = entire_classes - zero_classes  # 집합 차집합 연산

# 데이터 불러오기
print("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test = load_raw()

# 클래스 수 다시 설정
NB_CLASSES = y_train.shape[1]

K.set_image_data_format('channels_first')

# 모델 빌드 및 학습
print("Building and training DF model")
model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
print("Model compiled")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=NB_EPOCH,
    verbose=VERBOSE,
    validation_data=(X_valid, y_valid)
)

# 모델 저장
os.makedirs("models", exist_ok=True)
model_path = f'models/DF_{feature}_batch{BATCH_SIZE}_epoch{NB_EPOCH}.h5'
model.save(model_path)

# 평가
score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
y_pred = model.predict(X_test)

print("Testing accuracy:", score_test[1])
print("Testing loss:", score_test[0])

# 시각화
plot_corr(y_test, y_pred)
plot_history(history, score_test)

# 정확도 CSV 저장
y_true_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

report_dict = classification_report(y_true_labels, y_pred_labels, output_dict=True)
class_accuracies = {
    int(label): metrics['recall'] 
    for label, metrics in report_dict.items() 
    if label.isdigit()
}

df = pd.DataFrame(list(class_accuracies.items()), columns=["Class", "Accuracy"])
df = df.sort_values("Class")

os.makedirs("results", exist_ok=True)
csv_path = f"results/DF_{feature}_batch{BATCH_SIZE}_epoch{NB_EPOCH}.csv"
df.to_csv(csv_path, index=False)

print(f"클래스별 분류 정확도 CSV 파일로 저장 완료: {csv_path}")

# 샘플별 예측 확률, 예측 클래스, 실제 라벨 저장
y_true_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

pred_with_info = np.hstack((
    y_pred,
    y_pred_labels.reshape(-1, 1),
    y_true_labels.reshape(-1, 1)
))

class_columns = [f"class_{i}_prob" for i in range(NB_CLASSES)]
columns = class_columns + ["predicted_class", "true_label"]

df_pred = pd.DataFrame(pred_with_info, columns=columns)
pred_csv_path = f"results/DF_{feature}_predictions_per_sample.csv"
df_pred.to_csv(pred_csv_path, index=False)

print(f"테스트 샘플별 예측 확률, 예측 클래스, 실제 라벨 저장 완료: {pred_csv_path}")
