#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
import random
import logging
import numpy as np
import pandas as pd
from keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from data_loader import load_raw, load_npz
from plot_result import plot_corr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set stdout encoding for consistent output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set random seed for reproducibility
random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def test_model(model_path, dataset_path, feature="direction", nb_classes=75, length=5000, verbose=1):
    logger.info(f"Testing model from {model_path} with dataset from {dataset_path}")

    # Configuration
    FEATURE = feature
    NB_CLASSES = nb_classes
    LENGTH = length
    VERBOSE = verbose
    INPUT_SHAPE = (LENGTH, 1)

    # Load and prepare data
    logger.info("Loading and preparing data for model testing")
    _, _, _, _, X_test, y_test = load_npz(dataset_path, FEATURE)

    K.set_image_data_format('channels_first')

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # Evaluate model
    score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
    y_pred = model.predict(X_test)

    logger.info(f"Testing accuracy: {score_test[1]:.4f}")
    logger.info(f"Testing loss: {score_test[0]:.4f}")

    # Visualize results
    plot_corr(y_test, y_pred)

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
    
    # Extract model name for CSV path
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    csv_path = f"results/{model_name}_firefox_fiber_nodef_wtfpad_{FEATURE}_test_accuracy.csv"
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
    pred_csv_path = f"results/{model_name}_firefox_fiber_nodef_wtfpad_{FEATURE}_predictions_per_sample.csv"
    df_pred.to_csv(pred_csv_path, index=False)
    logger.info(f"Prediction probabilities, predicted class, and true labels per test sample saved to {pred_csv_path}")

if __name__ == "__main__":
    test_model_path = '/home/kwakrhkr59/XAI_WF/attack/df/models/DF_firefox_fiber_nodef_direction_batch128_epoch30.h5'
    test_dataset_path = '/home/kwakrhkr59/XAI_WF/dataset/wtfpad/firefox_fiber_direction_all.npz'

    test_model(test_model_path, test_dataset_path)