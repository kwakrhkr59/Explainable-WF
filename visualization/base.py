import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class Config:
    def __init__(self):
        self.xlim = (0, 10)
        self.ylim = (-3000, 300)
        self.zero_start = False
        self.class_name = "firefox_fiber"
        self.data_dir = self._set_data_dir()
        self.image_size = (250, 250)
        self.batch_size = 32
        self.epochs = 10
        self.model_save_path = self._set_model_save_path()
        self.history_save_path = self._set_history_save_path()
        self.gradcam_output_dir = self._set_gradcam_output_dir()
        self.desired_samples_per_class = 10

    def _set_data_dir(self):
        if self.zero_start:
            return f"/home/kwakrhkr59/XAI_StarPrint/data/{self.class_name}_{self.ylim[0]}_{self.ylim[1]}_{self.xlim[1]}_zero/"
        else:
            return f"/home/kwakrhkr59/XAI_StarPrint/data/{self.class_name}_{self.ylim[0]}_{self.ylim[1]}_{self.xlim[1]}/"

    def _set_model_save_path(self):
        base_path = f"model/{self.class_name}_{self.ylim[0]}_{self.ylim[1]}_{self.xlim[1]}"
        return f"{base_path}_zero_epoch{self.epochs}.h5" if self.zero_start else f"{base_path}_epoch{self.epochs}.h5"

    def _set_history_save_path(self):
        base_path = f"history/{self.class_name}_{self.ylim[0]}_{self.ylim[1]}_{self.xlim[1]}"
        return f"{base_path}_zero_epoch{self.epochs}.png" if self.zero_start else f"{base_path}_epoch{self.epochs}.png"

    def _set_gradcam_output_dir(self):
        base_path = f"results/{self.class_name}_{self.ylim[0]}_{self.ylim[1]}_{self.xlim[1]}"
        return f"{base_path}_zero_epoch{self.epochs}_per_class/" if self.zero_start else f"{base_path}_epoch{self.epochs}_per_class/"


# --- Custom Callback for Epoch Logging ---
class EpochLogger(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}/{self.epochs} - "
              f"loss: {logs.get('loss'):.4f} - "
              f"accuracy: {logs.get('accuracy'):.4f} - "
              f"val_loss: {logs.get('val_loss'):.4f} - "
              f"val_accuracy: {logs.get('val_accuracy'):.4f}")


# --- Data Loading and Preprocessing ---
def load_images_from_directory(data_dir, image_size):
    """
    Loads images and labels from a structured directory.

    Args:
        data_dir (str): Path to the root directory containing class subdirectories.
        image_size (tuple): Desired size (height, width) for resizing images.

    Returns:
        tuple: A tuple containing:
            - np.array: Array of image data.
            - np.array: Array of corresponding labels.
            - list: List of class names.
    """
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    valid_class_names = []

    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        valid_class_names.append(class_name) # Only add if it's a valid directory
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(".png")]

        for image_file in image_files:
            image = tf.keras.preprocessing.image.load_img(image_file, target_size=image_size)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_paths.append(image_array)
            labels.append(label_idx) # Use label_idx for numerical labels

    return np.array(image_paths), np.array(labels), valid_class_names


def setup_data_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """
    Sets up ImageDataGenerators for training, validation, and testing.

    Args:
        X_train (np.array): Training image data.
        y_train (np.array): Training labels.
        X_val (np.array): Validation image data.
        y_val (np.array): Validation labels.
        X_test (np.array): Test image data.
        y_test (np.array): Test labels.
        batch_size (int): Batch size for the generators.

    Returns:
        tuple: A tuple containing train_generator, val_generator, test_generator.
    """
    datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = datagen.flow(X_val, y_val, batch_size=batch_size)
    test_generator = datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)
    return train_generator, val_generator, test_generator


# --- Model Definition ---
def build_cnn_model(image_size, num_classes):
    """
    Builds a Convolutional Neural Network model.

    Args:
        image_size (tuple): Input image size (height, width).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
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
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# --- History Plotting ---
def plot_and_save_history(history, save_path, epochs, config):
    """
    Plots training and validation accuracy/loss and saves the plot.

    Args:
        history (tf.keras.callbacks.History): History object returned by model.fit.
        save_path (str): Path to save the history plot.
        epochs (int): Number of epochs.
        config (Config): Configuration object for file naming.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"History plot saved to {save_path}")


# --- Grad-CAM Functions ---
def find_last_conv_layer(model):
    """
    Finds the name of the last convolutional layer in the model.

    Args:
        model (tf.keras.Model): The Keras model.

    Returns:
        str: Name of the last convolutional layer.

    Raises:
        ValueError: If no suitable Conv2D layer is found.
    """
    for layer in reversed(model.layers):
        if "conv2d" in layer.name:
            return layer.name
    raise ValueError("No suitable Conv2D layer found for Grad-CAM. Check your model architecture.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image.

    Args:
        img_array (np.array): Preprocessed image array (e.g., scaled to [0, 1]).
        model (tf.keras.Model): The trained Keras model.
        last_conv_layer_name (str): Name of the last convolutional layer.
        pred_index (int, optional): Index of the class to target. If None, uses the
                                     predicted class. Defaults to None.

    Returns:
        np.array: The generated heatmap.
    """
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


def display_gradcam(image_array_raw, heatmap, image_size, alpha=0.4):
    """
    Superimposes the heatmap on the original image.

    Args:
        image_array_raw (np.array): Original, unnormalized image array.
        heatmap (np.array): The Grad-CAM heatmap.
        image_size (tuple): The target image size (height, width) for display.
        alpha (float, optional): Transparency factor for the heatmap. Defaults to 0.4.

    Returns:
        np.array: The superimposed image.
    """
    # Convert image_array_raw to uint8 if not already
    img = image_array_raw.astype(np.uint8)

    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed_img = heatmap * alpha + img
    superimposed_img = cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return superimposed_img


def generate_gradcam_visualizations(
    model,
    X_test,
    y_test,
    class_names,
    last_conv_layer_name,
    config
):
    """
    Generates and saves Grad-CAM visualizations for test samples.

    Args:
        model (tf.keras.Model): The trained Keras model.
        X_test (np.array): Test image data.
        y_test (np.array): Test labels.
        class_names (list): List of class names.
        last_conv_layer_name (str): Name of the last convolutional layer for Grad-CAM.
        config (Config): Configuration object.
    """
    os.makedirs(config.gradcam_output_dir, exist_ok=True)

    for class_idx, class_name in enumerate(class_names):
        print(f"\nProcessing Grad-CAM for class: {class_name}")

        class_indices = np.where(y_test == class_idx)[0]
        num_samples_to_plot = min(config.desired_samples_per_class, len(class_indices))

        if num_samples_to_plot == 0:
            print(f"No samples found for class: {class_name} in test set. Skipping Grad-CAM.")
            continue

        np.random.shuffle(class_indices)

        fig, axes = plt.subplots(2, num_samples_to_plot, figsize=(3 * num_samples_to_plot, 6))
        fig.suptitle(f"Grad-CAM for Class: {class_name} (Ground Truth)", fontsize=16)

        # Handle case with single sample to ensure axes is 2D
        if num_samples_to_plot == 1:
            axes = np.expand_dims(axes, axis=1)

        for col_idx in range(num_samples_to_plot):
            i = class_indices[col_idx]
            
            # Original image for display and Grad-CAM input (normalized)
            img_array_raw = X_test[i]
            img_array_preprocessed = np.expand_dims(img_array_raw / 255.0, axis=0) # Normalize for model input

            true_label_idx = y_test[i]
            
            preds = model.predict(img_array_preprocessed, verbose=0)
            predicted_label_idx = np.argmax(preds[0])

            heatmap = make_gradcam_heatmap(img_array_preprocessed, model, last_conv_layer_name, pred_index=predicted_label_idx)
            gradcam_image = display_gradcam(img_array_raw, heatmap, config.image_size)
            
            # Display original image
            axes[0, col_idx].imshow(img_array_raw.astype(np.uint8))
            axes[0, col_idx].set_title(f"Pred: {class_names[predicted_label_idx]}")
            axes[0, col_idx].axis('off')

            # Display Grad-CAM image
            axes[1, col_idx].imshow(gradcam_image)
            axes[1, col_idx].set_title("Grad-CAM")
            axes[1, col_idx].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(config.gradcam_output_dir, f"GradCAM_Class_{class_name}_Samples.png")
        plt.savefig(save_path)
        plt.close(fig)
    print(f"\nGrad-CAM visualizations saved to {config.gradcam_output_dir}")


# --- Main Execution Flow ---
def main():
    config = Config()

    print(f"Loading images from: {config.data_dir}")
    images, labels, class_names = load_images_from_directory(config.data_dir, config.image_size)
    print(f"Found {len(images)} images belonging to {len(class_names)} classes.")

    if len(images) == 0:
        print("No images found. Exiting.")
        return

    # Data Splitting
    # Ensure stratification works even if a class has only one sample
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    except ValueError as e:
        print(f"Error during data splitting: {e}. This often happens if a class has only one sample.")
        print("Attempting to proceed without stratification for temp split if error is due to single sample class.")
        X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    # Setup Data Generators
    train_generator, val_generator, test_generator = setup_data_generators(
        X_train, y_train, X_val, y_val, X_test, y_test, config.batch_size
    )

    # Build and Compile Model
    model = build_cnn_model(config.image_size, len(class_names))
    model.summary()

    # Train Model
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=[EpochLogger(config.epochs)]
    )
    print("Model training finished.")

    # Evaluate Model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save Model
    os.makedirs("model", exist_ok=True)
    model.save(config.model_save_path)
    print(f"Model saved to {config.model_save_path}")

    # Draw and Save History
    os.makedirs("history", exist_ok=True)
    plot_and_save_history(history, config.history_save_path, config.epochs, config)

    # Grad-CAM Visualization
    print("\nGenerating Grad-CAM visualizations...")
    last_conv_layer_name = find_last_conv_layer(model)
    print(f"Using '{last_conv_layer_name}' as the last convolutional layer for Grad-CAM.")

    generate_gradcam_visualizations(
        model,
        X_test,
        y_test,
        class_names,
        last_conv_layer_name,
        config
    )
    print("Grad-CAM visualization generation complete.")

if __name__ == "__main__":
    main()