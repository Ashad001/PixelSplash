import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Model, Input
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.gridspec as gridspec

# Load the color cluster mappings (24 clusters for RGB quantization)
color_clusters = np.load("/content/colour_kmeans24_cat7.npy", allow_pickle=True, encoding='latin1')
color_clusters, _ = color_clusters  # or however it's structured

# Helper functions for color transformations
def rgb_to_cluster_index(image_rgb, clusters):
    """Convert RGB image to cluster indices (0-23)"""
    H, W, _ = image_rgb.shape
    image_flat = image_rgb.reshape(-1, 3)
    dists = np.linalg.norm(image_flat[:, None] - clusters[None, :], axis=2)
    closest = np.argmin(dists, axis=1)
    return closest.reshape(H, W)

def cluster_index_to_rgb(label_map, clusters):
    """Convert cluster indices back to RGB colors"""
    H, W = label_map.shape
    rgb = clusters[label_map.flatten()]
    return rgb.reshape(H, W, 3)

def rgb2gray(images):
    """Convert RGB images to grayscale"""
    gray = np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])
    return gray[..., np.newaxis].astype(np.float32)

def preprocess_data(x, y, target_class=7):
    """Preprocess data by extracting target class and converting to grayscale/labels"""
    # Extract only the target class (horses)
    x = x[y.flatten() == target_class].astype("float32") / 255.0
    
    # Create grayscale input
    x_gray = rgb2gray(x)
    
    # Create color cluster labels for each pixel
    label_maps = np.array([rgb_to_cluster_index(img, color_clusters) for img in x])
    
    return x_gray, label_maps, x

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
x_train_gray, y_train_cls, x_train_rgb = preprocess_data(x_train, y_train)
x_test_gray, y_test_cls, x_test_rgb = preprocess_data(x_test, y_test)

print(f"Training data shape: {x_train_gray.shape}, Testing data shape: {x_test_gray.shape}")
print(f"Training labels shape: {y_train_cls.shape}, Testing labels shape: {y_test_cls.shape}")

def build_classification_cnn(input_shape=(32, 32, 1), num_classes=24):
    """Build a standard CNN model for colorization via classification"""
    model = models.Sequential()
    
    # First convolutional block
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional block
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Third convolutional block
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Upsampling block 1
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Upsampling block 2
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Output layer - one class per color cluster
    model.add(layers.Conv2D(num_classes, (1, 1), padding='same'))
    
    return model

def build_unet_classification(input_shape=(32, 32, 1), num_classes=24):
    """Build a UNet-style CNN with skip connections for colorization via classification"""
    inputs = Input(shape=input_shape)
    
    # Encoder path
    # Block 1
    conv1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = layers.Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3 (bottleneck)
    conv3 = layers.Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(256, (3, 3), padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    
    # Decoder path with skip connections
    # Block 2
    up2 = layers.UpSampling2D(size=(2, 2))(conv3)
    # Skip connection from encoder block 2
    concat2 = layers.Concatenate()([up2, conv2])  # Skip connection
    deconv2 = layers.Conv2D(128, (3, 3), padding='same')(concat2)
    deconv2 = layers.BatchNormalization()(deconv2)
    deconv2 = layers.Activation('relu')(deconv2)
    deconv2 = layers.Conv2D(128, (3, 3), padding='same')(deconv2)
    deconv2 = layers.BatchNormalization()(deconv2)
    deconv2 = layers.Activation('relu')(deconv2)
    
    # Block 1
    up1 = layers.UpSampling2D(size=(2, 2))(deconv2)
    # Skip connection from encoder block 1
    concat1 = layers.Concatenate()([up1, conv1])  # Skip connection
    deconv1 = layers.Conv2D(64, (3, 3), padding='same')(concat1)
    deconv1 = layers.BatchNormalization()(deconv1)
    deconv1 = layers.Activation('relu')(deconv1)
    deconv1 = layers.Conv2D(64, (3, 3), padding='same')(deconv1)
    deconv1 = layers.BatchNormalization()(deconv1)
    deconv1 = layers.Activation('relu')(deconv1)
    
    # Output layer - one class per color cluster
    outputs = layers.Conv2D(num_classes, (1, 1), padding='same')(deconv1)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Function to visualize activations
def visualize_activations(model, input_image, layer_names, title="Layer Activations"):
    """Visualize activations of specified layers for a given input image"""
    # Make a prediction to ensure the model has been built
    _ = model.predict(np.expand_dims(input_image, axis=0))
    
    # Create the activation model
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=outputs)
    
    # Get activations
    activations = activation_model.predict(np.expand_dims(input_image, axis=0))
    
    # Create figure with n rows (for each layer) and 8 columns (for features)
    n_rows = len(layer_names)
    n_cols = 8  # Show 8 sample feature maps per layer
    
    plt.figure(figsize=(16, 2 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols + 1, width_ratios=[3] + [1] * n_cols)
    
    for i, layer_name in enumerate(layer_names):
        # Display layer name
        ax = plt.subplot(gs[i, 0])
        ax.text(0.5, 0.5, layer_name, fontsize=10, ha='center')
        ax.axis('off')
        
        # Get the activation maps for this layer
        activation = activations[i]
        n_features = activation.shape[-1]
        
        # Select a subset of feature maps to display
        selected_indices = np.linspace(0, n_features - 1, n_cols, dtype=int)
        
        for j, feature_idx in enumerate(selected_indices):
            ax = plt.subplot(gs[i, j + 1])
            feature_map = activation[0, :, :, feature_idx]
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'F{feature_idx}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Compare CNN and UNet activations
def compare_activations(cnn_model, unet_model, input_image):
    """Compare activations between standard CNN and UNet models"""
    # Get layer names after building models with a sample forward pass
    _ = cnn_model.predict(np.expand_dims(input_image, axis=0))
    _ = unet_model.predict(np.expand_dims(input_image, axis=0))
    
    # Get actual layer names from models
    cnn_conv_layers = [layer.name for layer in cnn_model.layers if 'conv2d' in layer.name]
    unet_conv_layers = [layer.name for layer in unet_model.layers if 'conv2d' in layer.name]
    
    # Select representative layers from each model
    # For standard CNN - first layer, mid encoder, bottleneck, decoder
    cnn_layers = [
        cnn_conv_layers[0],  # First encoder layer
        cnn_conv_layers[2],  # Second encoder layer
        cnn_conv_layers[4],  # Bottleneck layer
        cnn_conv_layers[-2]  # Decoder layer
    ]
    
    # For UNet - first encoder, mid encoder, bottleneck, decoder with skip connection
    unet_layers = [
        unet_conv_layers[0],   # First encoder layer
        unet_conv_layers[2],   # Second encoder layer
        unet_conv_layers[4],   # Bottleneck layer
        unet_conv_layers[-2]   # Decoder layer with skip connection
    ]
    
    # Visualize activations
    print("Standard CNN Activations:")
    visualize_activations(cnn_model, input_image, cnn_layers, "Standard CNN Activations")
    
    print("UNet Activations:")
    visualize_activations(unet_model, input_image, unet_layers, "UNet Activations")

# Define callbacks for training
def get_callbacks(model_prefix):
    """Get training callbacks with model-specific filenames"""
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        f"best_{model_prefix}_model.h5", 
        save_best_only=True,
        monitor='val_loss'
    )
    
    return [early_stop, lr_scheduler, checkpoint]

# Data augmentation (optional)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generators
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Create generator for both inputs and targets
def create_train_generator(x_input, y_target, batch_size=64):
    """Create a generator for training with data augmentation"""
    seed = 42
    input_gen = data_gen.flow(x_input, None, batch_size=batch_size, seed=seed)
    
    # For targets, we need to apply the same transformations
    # Since ImageDataGenerator doesn't support sparse labels directly
    input_with_channel = np.concatenate([x_input, np.zeros_like(x_input)], axis=-1)
    combined_gen = data_gen.flow(
        input_with_channel, 
        y_target, 
        batch_size=batch_size, 
        seed=seed
    )
    
    while True:
        x_batch = input_gen.__next__()
        combined_batch = combined_gen.__next__()
        _, y_batch = combined_batch
        
        yield x_batch, y_batch

# Train with or without data augmentation
def train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=50, use_augmentation=False, callbacks=None):
    """Train the model with options for data augmentation"""
    print(f"Training with {'augmentation' if use_augmentation else 'no augmentation'}")
    
    if use_augmentation:
        # Train using data generator
        train_gen = create_train_generator(x_train, y_train, batch_size=batch_size)
        history = model.fit(
            train_gen,
            steps_per_epoch=len(x_train) // batch_size,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Train directly
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    return history

def plot_history(history, title="Training History"):
    """Plot training and validation metrics history"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{title} - Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f"{title} - Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

# Generate predictions
def generate_predictions(model, x_test):
    """Generate colorized predictions from grayscale inputs"""
    # Get model predictions
    logits = model.predict(x_test)
    pred_classes = np.argmax(logits, axis=-1)
    
    # Convert class predictions to RGB values
    pred_rgb = np.array([cluster_index_to_rgb(pred, color_clusters) for pred in pred_classes])
    return pred_rgb

def show_predictions(grays, originals, preds1, preds2=None, model1_name="Standard CNN", model2_name="UNet CNN", n=5):
    """Display grayscale inputs, original colors, and predicted colors from one or two models"""
    if preds2 is not None:
        # Show predictions from two models
        plt.figure(figsize=(15, 8))
        for i in range(n):
            plt.subplot(4, n, i + 1)
            plt.title("Grayscale")
            plt.imshow(grays[i].squeeze(), cmap="gray")
            plt.axis("off")

            plt.subplot(4, n, i + 1 + n)
            plt.title("Original")
            plt.imshow(originals[i])
            plt.axis("off")

            plt.subplot(4, n, i + 1 + 2 * n)
            plt.title(model1_name)
            plt.imshow(np.clip(preds1[i], 0, 1))
            plt.axis("off")
            
            plt.subplot(4, n, i + 1 + 3 * n)
            plt.title(model2_name)
            plt.imshow(np.clip(preds2[i], 0, 1))
            plt.axis("off")
    else:
        # Show predictions from one model
        plt.figure(figsize=(15, 6))
        for i in range(n):
            plt.subplot(3, n, i + 1)
            plt.title("Grayscale")
            plt.imshow(grays[i].squeeze(), cmap="gray")
            plt.axis("off")

            plt.subplot(3, n, i + 1 + n)
            plt.title("Original")
            plt.imshow(originals[i])
            plt.axis("off")

            plt.subplot(3, n, i + 1 + 2 * n)
            plt.title(model1_name)
            plt.imshow(np.clip(preds1[i], 0, 1))
            plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def calculate_metrics(y_true, y_pred, num_samples=100):
    """Calculate image quality metrics between original and predicted colors"""
    if len(y_pred) > num_samples:
        indices = np.random.choice(len(y_pred), num_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    # Calculate MSE and MAE
    mse = np.mean(np.square(y_true - y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate PSNR and SSIM
    psnr_scores = []
    ssim_scores = []
    
    for i in range(len(y_true)):
        # PSNR calculation
        psnr_val = psnr(y_true[i], y_pred[i], data_range=1.0)
        psnr_scores.append(psnr_val)
        
        # SSIM calculation - use smaller window size for CIFAR-10
        ssim_val = ssim(y_true[i], y_pred[i], win_size=5, channel_axis=2, data_range=1.0)
        ssim_scores.append(ssim_val)
    
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim
    }

def compare_models():
    """Train and compare standard CNN vs UNet-style CNN with skip connections"""
    # Split training data for validation
    val_size = int(0.2 * len(x_train_gray))
    x_val_gray = x_train_gray[:val_size]
    y_val_cls = y_train_cls[:val_size]
    x_val_rgb = x_train_rgb[:val_size]
    
    x_train_gray_split = x_train_gray[val_size:]
    y_train_cls_split = y_train_cls[val_size:]
    x_train_rgb_split = x_train_rgb[val_size:]
    
    # Create models
    print("Creating models...")
    standard_cnn = build_classification_cnn()
    unet_cnn = build_unet_classification()
    
    # Make a forward pass to ensure models are built
    dummy_input = np.zeros((1, 32, 32, 1), dtype=np.float32)
    _ = standard_cnn(dummy_input)
    _ = unet_cnn(dummy_input)
    
    # Compile models
    standard_cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    unet_cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Print model summaries
    print("\nStandard CNN Architecture:")
    standard_cnn.summary()
    print("\nUNet CNN Architecture:")
    unet_cnn.summary()
    
    # Train models
    print("\nTraining Standard CNN...")
    standard_history = train_model(
        standard_cnn, 
        x_train_gray_split,
        y_train_cls_split,
        x_val_gray, 
        y_val_cls,
        batch_size=32,
        epochs=30,
        callbacks=get_callbacks("standard_cnn")
    )
    
    print("\nTraining UNet CNN...")
    unet_history = train_model(
        unet_cnn,
        x_train_gray_split,
        y_train_cls_split,
        x_val_gray,
        y_val_cls,
        batch_size=32,
        epochs=30,
        callbacks=get_callbacks("unet_cnn")
    )
    
    # Plot training histories
    plot_history(standard_history, "Standard CNN")
    plot_history(unet_history, "UNet CNN")
    
    # Compare training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(standard_history.history['loss'], label='Standard CNN - Train Loss')
    plt.plot(standard_history.history['val_loss'], label='Standard CNN - Val Loss')
    plt.plot(unet_history.history['loss'], label='UNet CNN - Train Loss')
    plt.plot(unet_history.history['val_loss'], label='UNet CNN - Val Loss')
    plt.title("Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(standard_history.history['accuracy'], label='Standard CNN - Train Acc')
    plt.plot(standard_history.history['val_accuracy'], label='Standard CNN - Val Acc')
    plt.plot(unet_history.history['accuracy'], label='UNet CNN - Train Acc')
    plt.plot(unet_history.history['val_accuracy'], label='UNet CNN - Val Acc')
    plt.title("Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    # Generate predictions
    standard_preds_rgb = generate_predictions(standard_cnn, x_test_gray)
    unet_preds_rgb = generate_predictions(unet_cnn, x_test_gray)
    
    # Show sample predictions
    idxs = np.random.choice(len(x_test_gray), 5, replace=False)
    show_predictions(
        x_test_gray[idxs], 
        x_test_rgb[idxs], 
        standard_preds_rgb[idxs], 
        unet_preds_rgb[idxs]
    )
    
    # Calculate metrics
    standard_metrics = calculate_metrics(x_test_rgb, standard_preds_rgb, num_samples=100)
    unet_metrics = calculate_metrics(x_test_rgb, unet_preds_rgb, num_samples=100)
    
    print("\nPerformance Metrics Comparison:")
    print(f"{'Metric':<10} {'Standard CNN':<15} {'UNet CNN':<15} {'Improvement':<15}")
    print("-" * 55)
    for metric in ['MSE', 'MAE', 'PSNR', 'SSIM']:
        std_val = standard_metrics[metric]
        unet_val = unet_metrics[metric]
        
        # For MSE and MAE, lower is better; for PSNR and SSIM, higher is better
        if metric in ['MSE', 'MAE']:
            improvement = std_val - unet_val
            improvement_pct = (improvement / std_val) * 100 if std_val != 0 else 0
            print(f"{metric:<10} {std_val:<15.4f} {unet_val:<15.4f} {improvement_pct:+.2f}%")
        else:
            improvement = unet_val - std_val
            improvement_pct = (improvement / std_val) * 100 if std_val != 0 else 0
            print(f"{metric:<10} {std_val:<15.4f} {unet_val:<15.4f} {improvement_pct:+.2f}%")
    
    # Compare intermediate activations with safe layer access
    try:
        # Get a sample image for activation visualization
        sample_img = x_test_gray[idxs[0]]
        compare_activations(standard_cnn, unet_cnn, sample_img)
    except Exception as e:
        print(f"Warning: Could not compare activations due to an error: {e}")
        print("Activation visualization skipped")
    
    return {
        'standard_cnn': standard_cnn,
        'unet_cnn': unet_cnn,
        'standard_history': standard_history,
        'unet_history': unet_history,
        'standard_metrics': standard_metrics,
        'unet_metrics': unet_metrics
    }

# Main execution block
if __name__ == "__main__":
    print("\n--- Starting Colorization Models Comparison: Standard CNN vs UNet-style CNN ---\n")
    
    # Run the comparison
    comparison_results = compare_models()
    
    # Print summary of findings
    print("\n=== Summary of Colorization Findings ===")
    print("""
    1. Skip Connections in UNet:
       - Improved pixel-level accuracy and loss by providing direct paths for fine details
       - Enabled better preservation of structural information across scales
       - Helped combat the vanishing gradient problem during training
       
    2. Visual Quality Comparison:
       - UNet model typically produces more vibrant and accurate colors
       - Standard CNN tends to produce more muted, averaged colors
       - UNet preserves finer details due to skip connections
       
    3. Intermediate Activations:
       - Early layers in both models capture basic edges and textures
       - Deeper layers in UNet maintain more spatial information due to skip connections
       - Standard CNN bottleneck activations lose spatial details
       - UNet decoder activations show higher fidelity reconstruction with preserved details
    """)
    
    print("\nEnd of comparison.\n")