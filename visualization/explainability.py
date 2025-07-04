from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import os

import shap
import cv2

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import itertools

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

from exmatchina import ExMatchina

# --- 전역 변수 및 헬퍼 함수 ---
num_classes = 95
classes = ['Non-ecotic beats (normal beat)',
           'Supraventricular ectopic beats',
           'Ventricular ectopic beats',
           'Fusion beats',
           'Unknown beats']
class_dict = {
    'Non-ecotic beats (normal beat)': 0,
    'Supraventricular ectopic beats': 1,
    'Ventricular ectopic beats': 2,
    'Fusion beats': 3,
    'Unknown beats': 4
}
inv_class_dict = {v: k for k, v in class_dict.items()}

# --- 시각화 헬퍼 함수 ---

def plot_shap_with_timestamps(shap_values_data, original_data_with_timestamps):
    timestamp_values = original_data_with_timestamps[:, 0]
    direction_values = original_data_with_timestamps[:, 1]

    colors = []
    for l in np.linspace(1,0,100):
        colors.append((0/255,255/255,255/255,l))
    for l in np.linspace(0,1,100):
        colors.append((255/255,0/255,255/255,l))
    cmm = matplotlib.colors.LinearSegmentedColormap.from_list("shap", colors)

    fig, ax = plt.subplots(ncols=2, figsize=(18,6))

    ax[0].plot(timestamp_values, direction_values, color='black')
    ax[0].set_xlabel('Timestamp')
    ax[0].set_ylabel('Direction Value')
    ax[0].set_title('Original Signal')

    ax[1].plot(timestamp_values, direction_values, color='black')

    gradient = shap_values_data.flatten()
    max_g = np.amax(gradient)
    min_g = np.amin(gradient)
    
    if (max_g - min_g) == 0:
        gradient = np.zeros_like(gradient)
    else:
        gradient = (gradient - min_g) / (max_g - min_g)

    for j in range(len(gradient) - 1):
        current_color = cmm(gradient[j])
        ax[1].axvspan(timestamp_values[j], timestamp_values[j+1], facecolor=current_color, alpha=0.7)
    
    if len(gradient) > 0 and len(timestamp_values) > 1:
        current_color_last = cmm(gradient[-1])
        approx_interval = timestamp_values[-1] - timestamp_values[-2]
        ax[1].axvspan(timestamp_values[-1], timestamp_values[-1] + approx_interval, facecolor=current_color_last, alpha=0.7)

    ax[1].set_xlabel('Timestamp')
    ax[1].set_ylabel('Direction Value')
    ax[1].set_title('SHAP Values Overlay')

    sm = cm.ScalarMappable(cmap=cmm)
    sm.set_clim(min_g, max_g)

    cb = fig.colorbar(sm, ax=ax.ravel().tolist(), label="SHAP Value", aspect=60)
    cb.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig('results/nodef/shap.png')

def plot_map_with_timestamps(cls_pred, grads, original_data_with_timestamps, inv_class_dict):
    timestamp_values = original_data_with_timestamps[:, 0]
    direction_values = original_data_with_timestamps[:, 1]

    print(inv_class_dict[cls_pred])
    fig, ax = plt.subplots(ncols=2, figsize=(18,6))
    
    ax[0].plot(timestamp_values, direction_values, color='black')
    ax[0].set_xlabel('Timestamp')
    ax[0].set_ylabel('Direction Value')
    ax[0].set_title('Original Signal')

    ax[1].plot(timestamp_values, direction_values, color='black')

    gradient = grads.flatten()
    max_g = np.amax(gradient)
    min_g = np.amin(gradient)
    
    if (max_g - min_g) == 0:
        gradient = np.zeros_like(gradient)
    else:
        gradient = (gradient - min_g) / (max_g - min_g)

    for i in range(len(gradient) - 1):
        current_color = cm.cool(gradient[i])
        ax[1].axvspan(timestamp_values[i], timestamp_values[i+1], facecolor=current_color, alpha=0.7)
    
    if len(gradient) > 0 and len(timestamp_values) > 1:
        current_color_last = cm.cool(gradient[-1])
        approx_interval = timestamp_values[-1] - timestamp_values[-2]
        ax[1].axvspan(timestamp_values[-1], timestamp_values[-1] + approx_interval, facecolor=current_color_last, alpha=0.7)

    ax[1].set_xlabel('Timestamp')
    ax[1].set_ylabel('Direction Value')
    ax[1].set_title('Heatmap Overlay')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.12, 0.03, 0.75])
    sm = cm.ScalarMappable(cmap="cool")
    sm.set_clim(min_g, max_g)
    fig.colorbar(sm, cax=cbar_ax, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('results/nodef/saliency.png')

def plot_ecg_with_timestamps(test_data_with_timestamps, examples_data_with_timestamps, label, inv_class_dict):
    fig = plt.figure(figsize=(18,5))
    num_display = 4

    ax_test = fig.add_subplot(1, num_display, 1)
    ax_test.set_title(f'Test Signal: {inv_class_dict[label]}')
    ax_test.plot(test_data_with_timestamps[:, 0], test_data_with_timestamps[:, 1], color='black')
    ax_test.set_xlabel('Timestamp')
    ax_test.set_ylabel('Direction Value')

    for k in range(min(num_display - 1, len(examples_data_with_timestamps))):
        ax_example = fig.add_subplot(1, num_display, k + 2)
        example_data = examples_data_with_timestamps[k]
        ax_example.set_title(f'Example {k+1}: {inv_class_dict[label]}')
        ax_example.plot(example_data[:, 0], example_data[:, 1], color='#0067FF')
        ax_example.set_xlabel('Timestamp')
        ax_example.set_ylabel('Direction Value')

    fig.tight_layout()
    plt.savefig('results/nodef/3.png')

# --- main 함수 ---
def main():    
    data = np.load('/home/kwakrhkr59/XAI_WF/data/nodef/bigenough_original_all.npz')
    X_data = data['data']
    y_data = data['labels']
    
    num_samples = X_data.shape[0]
    signal_length = X_data.shape[1]

    for i in range(num_samples):
        X_data[i, :, 0] = np.cumsum(np.random.rand(signal_length) * np.random.uniform(0.01, 0.5, signal_length))
        X_data[i, :, 1] = np.sin(X_data[i, :, 0] / 5) * np.random.uniform(0.1, 2, signal_length) + np.random.randn(signal_length) * 0.1
    
    X_train = X_data[:int(num_samples*0.8), :, 1:2]
    y_train = to_categorical(y_data[:int(num_samples*0.8)], num_classes)

    x_test = X_data[int(num_samples*0.8):, :, 1:2]
    y_test = to_categorical(y_data[int(num_samples*0.8):], num_classes)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # 2. 모델 로드
    try:
        # model = load_model('/home/kwakrhkr59/XAI_WF/attack/df/models/DF_bigenough_nodef_direction_batch128_epoch30.h5')
        model = load_model('dummy')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Defining a dummy model for demonstration.")
        from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense, Activation
        from tensorflow.keras.optimizers import Adam
        
        inputs = Input(shape=(signal_length, 1), name='inputs_cnn')
        conv1 = Conv1D(filters=64, kernel_size=5, activation='relu', name='Conv1')(inputs)
        bn1 = BatchNormalization(name='Bn1')(conv1)
        mp1 = MaxPooling1D(pool_size=2, name='Mp1')(bn1)
        drop1 = Dropout(0.2, name='Drop1')(mp1)
        
        conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', name='Conv2')(drop1)
        bn2 = BatchNormalization(name='Bn2')(conv2)
        mp2 = MaxPooling1D(pool_size=2, name='Mp2')(bn2)
        drop2 = Dropout(0.2, name='Drop2')(mp2)
        
        conv3 = Conv1D(filters=64, kernel_size=3, activation='relu', name='Conv3')(drop2)
        bn3 = BatchNormalization(name='Bn3')(conv3)
        mp3 = MaxPooling1D(pool_size=2, name='Mp3')(bn3)
        drop3 = Dropout(0.2, name='Drop3')(mp3)
        
        flatten = Flatten()(drop3)
        dense1 = Dense(64, activation='relu', name='Dense1')(flatten)
        drop4 = Dropout(0.5, name='Drop4')(dense1)
        dense2 = Dense(32, activation='relu', name='Dense2')(drop4)
        logits = Dense(num_classes, name='logits')(dense2)
        probs = Activation('softmax', name='probs')(logits)

        model = Model(inputs=inputs, outputs=probs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Dummy model defined.")
    
    model.summary()

    # 3. 시각화할 샘플 인덱스 선택
    all_idx = np.random.choice(x_test.shape[0], 5, replace=False)
    print(f"Selected indices for visualization: {all_idx}")

    # 4. 각 해석 가능성 기법 적용 및 시각화
    print("\n--- SHAP Visualization ---")
    background_for_shap = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    
    def map2layer(x, layer_index):
        if x.ndim == 3:
            x_in = x
        elif x.ndim == 2:
            x_in = np.expand_dims(x, axis=0)
        else:
            raise ValueError(f'Unexpected input shape: {x.shape}')
        
        func = Model(inputs=model.input, outputs=model.layers[layer_index].input)
        return func.predict(x_in)

    for i in all_idx:
        print(f"\nProcessing SHAP for sample {i}")
        test_input = np.expand_dims(x_test[i], axis=0)
        original_data_for_plot = X_data[int(num_samples*0.8)+i]
        
        class_pred = np.argmax(model.predict(test_input))
        
        conv3_layer_index = None
        for idx, layer in enumerate(model.layers):
            if layer.name == 'Conv3':
                conv3_layer_index = idx
                break
        if conv3_layer_index is None:
            print("Conv3 layer not found. Skipping SHAP.")
            continue

        e = shap.GradientExplainer((model.layers[conv3_layer_index].input, model.layers[-1].output), map2layer(background_for_shap, conv3_layer_index))
        shap_values, indexes = e.shap_values(map2layer(test_input, conv3_layer_index), ranked_outputs=1)

        if isinstance(shap_values, list):
             shap_values = shap_values[0]

        if shap_values.ndim == 3:
            shap_values = shap_values.sum(axis=-1)
        
        original_feature_dim = shap_values.shape[1] if shap_values.ndim == 2 else shap_values.shape[0]
        if original_feature_dim != signal_length:
            interp_factor = signal_length / original_feature_dim
            new_shap_values = np.interp(np.arange(signal_length), 
                                         np.arange(original_feature_dim) * interp_factor, 
                                         shap_values.flatten())
            shap_values = new_shap_values.reshape(1, signal_length)
        
        plot_shap_with_timestamps(shap_values.squeeze(), original_data_for_plot)

    print("\n--- Saliency Map Visualization ---")
    for i in all_idx:
        print(f"\nProcessing Saliency Map for sample {i}")
        test_input = np.expand_dims(x_test[i], axis=0)
        original_data_for_plot = X_data[int(num_samples*0.8)+i]

        class_pred = np.argmax(model.predict(test_input))
        score = CategoricalScore([class_pred])
        saliency = Saliency(model, model_modifier=ReplaceToLinear())
        saliency_map = saliency(score, test_input)
        
        plot_map_with_timestamps(class_pred, saliency_map[0].squeeze(), original_data_for_plot, inv_class_dict)

    print("\n--- Grad-CAM++ Visualization ---")
    for i in all_idx:
        print(f"\nProcessing Grad-CAM++ for sample {i}")
        test_input = np.expand_dims(x_test[i], axis=0)
        original_data_for_plot = X_data[int(num_samples*0.8)+i]

        class_idx = np.argmax(model.predict(test_input))
        gradcampp_obj = GradcamPlusPlus(model, model_modifier=ReplaceToLinear())
        cam = gradcampp_obj(score, test_input, penultimate_layer='Conv3')
        
        plot_map_with_timestamps(class_idx, cam[0].squeeze(), original_data_for_plot, inv_class_dict)

    print("\n--- ExMatchina Visualization ---")
    selected_layer = 'Flatten' 
    exm = ExMatchina(model=model, layer=selected_layer, examples=X_train)
    
    for i in all_idx:
        print(f"\nProcessing ExMatchina for sample {i}")
        test_input = x_test[i]
        original_data_for_plot_test = X_data[int(num_samples*0.8)+i]
        
        label_for_exm = exm.get_label_for(test_input)
        (examples_for_exm, indices) = exm.return_nearest_examples(test_input)
        
        examples_with_timestamps_for_plot = []
        for j, ex in enumerate(examples_for_exm):
            original_example_data = X_data[indices[j]]
            examples_with_timestamps_for_plot.append(original_example_data)
        
        plot_ecg_with_timestamps(original_data_for_plot_test, examples_with_timestamps_for_plot, label_for_exm, inv_class_dict)

if __name__ == "__main__":
    main()