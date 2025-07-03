#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import classification_report
from Model_NoDef import DFNet
from plot_result import plot_history, plot_corr
from data_loader import load_raw, load_npz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set stdout encoding for consistent output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set random seed for reproducibility
random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

logger.info("Training and evaluating DF model for closed-world scenario on non-defended dataset")

# Configuration
FEATURE = "direction"
NB_CLASSES = 75
NB_EPOCH = 30
BATCH_SIZE = 128
VERBOSE = 1
LENGTH = 5000
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
INPUT_SHAPE = (LENGTH, 1)
# DATASET_DIR = '/home/kwakrhkr59/XAI_WF/defense/wtfpad/results/default_250626_163519'
DATASET_DIR = '/scratch4/starlink/WFdata75x80/firefox_fiber'
# DATASET_DIR = '/home/kwakrhkr59/XAI_WF/defense/wtfpad/results/bigenough_tor_fiber'
# DATASET_DIR = '/home/kwakrhkr59/XAI_WF/preprocessing/output/direction/bigenough_direction_all.npz'

# Load and prepare data
logger.info("Loading and preparing data for training and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test = load_raw(DATASET_DIR, FEATURE)

K.set_image_data_format('channels_first')

# Build and train model
logger.info("Building and training DF model")
model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
logger.info("Model compiled successfully")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=NB_EPOCH,
    verbose=VERBOSE,
    validation_data=(X_valid, y_valid)
)

# Save model
os.makedirs("models", exist_ok=True)
model_path = f'models/DF_firefox_fiber_nodef_{FEATURE}_batch{BATCH_SIZE}_epoch{NB_EPOCH}.h5'
model.save(model_path)
logger.info(f"Model saved to {model_path}")

# Evaluate model
score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
y_pred = model.predict(X_test)

logger.info(f"Testing accuracy: {score_test[1]:.4f}")
logger.info(f"Testing loss: {score_test[0]:.4f}")

# Visualize results
plot_corr(y_test, y_pred)
plot_history(history, score_test)

# Save class-wise accuracy to CSV
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
csv_path = f"results/DF_firefox_fiber_nodef_{FEATURE}_batch{BATCH_SIZE}_epoch{NB_EPOCH}.csv"
df_accuracy.to_csv(csv_path, index=False)
logger.info(f"Class-wise classification accuracy saved to {csv_path}")

# Save sample-wise prediction probabilities, predicted class, and true label
pred_with_info = np.hstack((
    y_pred,
    y_pred_labels.reshape(-1, 1),
    y_true_labels.reshape(-1, 1)
))

class_columns = [f"class_{i}_prob" for i in range(NB_CLASSES)]
columns = class_columns + ["predicted_class", "true_label"]

df_pred = pd.DataFrame(pred_with_info, columns=columns)
pred_csv_path = f"results/DF_firefox_fiber_nodef_{FEATURE}_predictions_per_sample.csv"
df_pred.to_csv(pred_csv_path, index=False)
logger.info(f"Prediction probabilities, predicted class, and true labels per test sample saved to {pred_csv_path}")