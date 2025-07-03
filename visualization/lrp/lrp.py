#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from keras import backend as K
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

import innvestigate
import innvestigate.utils as iutils

random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

description = "Training and evaluating DF model for closed-world scenario on non-defended dataset with LRP explanation"
print(description)

feature = "size"
NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 1
LENGTH = 5000
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
INPUT_SHAPE = (LENGTH, 1)

entire_classes = set(range(75))
zero_classes = set([])
excluded_classes = entire_classes - zero_classes

print("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test = load_raw()
NB_CLASSES = y_train.shape[1]

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

os.makedirs("models", exist_ok=True)
model_path = f'models/DF_{feature}_batch{BATCH_SIZE}_epoch{NB_EPOCH}.h5'
model.save(model_path)
print(f"Model saved to: {model_path}")

score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
y_pred = model.predict(X_test)

print(f"Testing accuracy: {score_test[1]:.4f}")
print(f"Testing loss: {score_test[0]:.4f}")

plot_history(history, score_test)

y_true_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

report_dict = classification_report(y_true_labels, y_pred_labels, output_dict=True)
class_accuracies = {
    int(label): metrics['recall']
    for label, metrics in report_dict.items()
    if label.isdigit()
}

df_accuracy = pd.DataFrame(list(class_accuracies.items()), columns=["Class", "Accuracy"])
df_accuracy = df_accuracy.sort_values("Class")

os.makedirs("results", exist_ok=True)
csv_path = f"results/DF_{feature}_batch{BATCH_SIZE}_epoch{NB_EPOCH}_class_accuracy.csv"
df_accuracy.to_csv(csv_path, index=False)
print(f"클래스별 분류 정확도 CSV 파일로 저장 완료: {csv_path}")

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

def apply_lrp_1d(model, data_array, pred_index=None, method="lrp.z"):
    model_wo_softmax = iutils.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer(method, model_wo_softmax, neuron_selection_mode="index")
    preds = model.predict(data_array)
    if pred_index is None:
        pred_index = np.argmax(preds[0])
    analysis = analyzer.analyze(data_array, neuron_selection=pred_index)
    lrp_relevance = np.squeeze(np.sum(np.abs(analysis[0]), axis=-1))
    max_relevance = lrp_relevance.max()
    if max_relevance > 0:
        lrp_relevance /= max_relevance
    return lrp_relevance

def plot_lrp_relevance(original_data, lrp_relevance, title="", output_path=None, relevance_threshold=0.7):
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(original_data, color='blue', label='Original Data', linewidth=1.5)
    highlight_indices = np.where(lrp_relevance > relevance_threshold)[0]
    if len(highlight_indices) > 0:
        breaks = np.where(np.diff(highlight_indices) != 1)[0] + 1
        segments = np.split(highlight_indices, breaks)
        first_segment = True
        for segment in segments:
            if len(segment) > 0:
                start_idx = segment[0]
                end_idx = segment[-1]
                plt.axvspan(start_idx, end_idx, color='red', alpha=0.3, 
                            label=f'High Relevance (>{relevance_threshold})' if first_segment else "")
                first_segment = False
    plt.title(f"{title}\nOriginal Data with LRP Highlights (Threshold: {relevance_threshold})", fontsize=14)
    plt.xlabel("Sequence Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.bar(range(len(lrp_relevance)), lrp_relevance, width=1.0, color='darkorange', alpha=0.8, label='LRP Relevance Score')
    plt.axhline(y=relevance_threshold, color='green', linestyle='--', label='Relevance Threshold', alpha=0.7)
    plt.title(f"LRP Relevance Score", fontsize=14)
    plt.xlabel("Sequence Index", fontsize=12)
    plt.ylabel("Relevance Score (0-1)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

lrp_output_dir = f"results_lrp/{feature}_batch{BATCH_SIZE}_epoch{NB_EPOCH}_per_class/"
os.makedirs(lrp_output_dir, exist_ok=True)

desired_samples_per_class = 5

print("\nGenerating LRP visualizations for test samples...")

try:
    pass
except NameError:
    class_names = [f"Class_{i}" for i in range(NB_CLASSES)]

for class_idx in range(NB_CLASSES):
    current_class_name = class_names[class_idx]
    print(f"\nProcessing LRP for class: {current_class_name}")
    class_indices = np.where(y_test[:, class_idx] == 1)[0]
    num_samples_to_plot = min(desired_samples_per_class, len(class_indices))
    if num_samples_to_plot == 0:
        print(f"No samples found for class: {current_class_name} in test set. Skipping LRP.")
        continue
    np.random.shuffle(class_indices)
    for sample_idx_in_class in range(num_samples_to_plot):
        i = class_indices[sample_idx_in_class]
        data_sample_for_lrp = np.expand_dims(X_test[i], axis=0)
        true_label_idx = np.argmax(y_test[i])
        preds = model.predict(data_sample_for_lrp)
        predicted_label_idx = np.argmax(preds[0])
        lrp_relevance_map = apply_lrp_1d(model, data_sample_for_lrp,
                                         pred_index=predicted_label_idx,
                                         method="lrp.z")
        sample_title = (f"{current_class_name} Sample {sample_idx_in_class+1}\n"
                        f"True: {class_names[true_label_idx]} / Pred: {class_names[predicted_label_idx]}")
        plot_lrp_relevance(np.squeeze(X_test[i]), lrp_relevance_map,
                           title=sample_title,
                           output_path=os.path.join(lrp_output_dir, f"LRP_Class_{class_idx}_Sample_{i}.png"))

print("LRP visualizations generation completed.")