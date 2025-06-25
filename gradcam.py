import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from natsort import natsorted, ns

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

xlim = (0, 10)
ylim = (-3000, 300)
zero_start = False
class_name = "firefox_fiber"
if zero_start: f"/home/kwakrhkr59/XAI_StarPrint/data/{class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}_zero/"
else: data_dir = f"/home/kwakrhkr59/XAI_StarPrint/data/{class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}/"

image_size = (250, 250)
batch_size = 32
epochs = 10

def load_images_from_directory(data_dir, image_size):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(".png")]

        for image_file in image_files:
            image = tf.keras.preprocessing.image.load_img(image_file, target_size=image_size)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_paths.append(image_array)
            labels.append(label)

    return np.array(image_paths), np.array(labels), class_names

class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"loss: {logs.get('loss'):.4f} - "
              f"accuracy: {logs.get('accuracy'):.4f} - "
              f"val_loss: {logs.get('val_loss'):.4f} - "
              f"val_accuracy: {logs.get('val_accuracy'):.4f}")

images, labels, class_names = load_images_from_directory(data_dir, image_size)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = datagen.flow(X_val, y_val, batch_size=batch_size)
test_generator = datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (5, 5), padding='same'),
    tf.keras.layers.Conv2D(64, (5, 5)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (7, 7), padding='same'),
    tf.keras.layers.Conv2D(128, (7, 7)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[EpochLogger()])

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

if zero_start: model.save(f"model/{class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}_zero_epoch{epochs}.h5")
else: model.save(f"model/{class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}_epoch{epochs}.h5")

def draw_history():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if zero_start: plt.savefig(f'history/{class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}_zero_epoch{epochs}.png')
    else: plt.savefig(f'history/{class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}_epoch{epochs}.png')

    plt.close()
    print(f"History saved as {class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}_epoch{epochs}.png")

draw_history()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image_path, heatmap, alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size[1], image_size[0]))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return superimposed_img

last_conv_layer_name = None
for layer in reversed(model.layers):
    if "conv2d" in layer.name:
        last_conv_layer_name = layer.name
        break
if last_conv_layer_name is None:
    raise ValueError("No suitable Conv2D layer found for Grad-CAM. Check your model architecture.")

print(f"Using '{last_conv_layer_name}' as the last convolutional layer for Grad-CAM.")

if zero_start: gradcam_output_dir = f"results/{class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}_zero_epoch{epochs}_per_class/"
else: gradcam_output_dir = f"results/{class_name}_{ylim[0]}_{ylim[1]}_{xlim[1]}_epoch{epochs}_per_class/"
os.makedirs(gradcam_output_dir, exist_ok=True)

desired_samples_per_class = 10 

for class_idx, class_name in enumerate(class_names):
    print(f"\nProcessing Grad-CAM for class: {class_name}")

    class_indices = np.where(y_test == class_idx)[0]
    num_samples_to_plot = min(desired_samples_per_class, len(class_indices))

    if num_samples_to_plot == 0:
        print(f"No samples found for class: {class_name} in test set. Skipping Grad-CAM.")
        continue

    np.random.shuffle(class_indices)

    fig, axes = plt.subplots(2, num_samples_to_plot, figsize=(3 * num_samples_to_plot, 6))
    fig.suptitle(f"Grad-CAM for Class: {class_name} (Ground Truth)", fontsize=16)

    if num_samples_to_plot == 1:
        axes = np.expand_dims(axes, axis=1)

    selected_samples_count = 0
    for col_idx in range(num_samples_to_plot):
        i = class_indices[col_idx]

        img_array = np.expand_dims(X_test[i], axis=0)
        true_label_idx = y_test[i]
        
        preds = model.predict(img_array / 255.0)
        predicted_label_idx = np.argmax(preds[0])

        heatmap = make_gradcam_heatmap(img_array / 255.0, model, last_conv_layer_name, pred_index=predicted_label_idx)

        temp_img_path = os.path.join(gradcam_output_dir, f"temp_image_{class_name}_{selected_samples_count}.png")
        tf.keras.preprocessing.image.save_img(temp_img_path, X_test[i].astype(np.uint8))

        gradcam_image = display_gradcam(temp_img_path, heatmap)
        
        axes[0, col_idx].imshow(X_test[i].astype(np.uint8))
        axes[0, col_idx].set_title(f"Pred: {class_names[predicted_label_idx]}")
        axes[0, col_idx].axis('off')

        axes[1, col_idx].imshow(gradcam_image)
        axes[1, col_idx].set_title("Grad-CAM")
        axes[1, col_idx].axis('off')
        
        os.remove(temp_img_path)
        selected_samples_count += 1
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(gradcam_output_dir, f"GradCAM_Class_{class_name}_Samples.png")
    plt.savefig(save_path)
    plt.close(fig)