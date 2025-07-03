# Original Pickel ver.
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os

LENGTH = 5000
n = 75 # number of outputs = number of classes = number of keywords 
INPUT_SHAPE = (LENGTH,1)

def plot_corr(y_test, y_pred, save_path='log/heatmap/default_heatmap.png'):
    if (y_test[0].shape[0] != 1): y_test = np.argmax(y_test, axis=1)
    if (y_pred[0].shape[0] != 1): y_pred = np.argmax(y_pred, axis=1)
    plt.clf()
    plt.figure(figsize=(20, 20))
    plt.title("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=False, center=0, cmap='coolwarm', fmt='g', cbar=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
def plot_history(history, score, save_path='log/default_history.png'):
    plt.clf()
    plt.figure(figsize=(12, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.suptitle(f'Test Accuracy: {score[1]:.4f}, Test Loss: {score[0]:.4f}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def preprocess_data(X, y, n=75):
    X = X.astype('float32')[:, :, np.newaxis]
    # y = np_utils.to_categorical(y, n)
    y = to_categorical(y, n)
    return X, y