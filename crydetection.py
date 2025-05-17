from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, InputLayer
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import librosa
import random
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
# Ensure GPU is being used
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# After GPU configuration
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# ======================
# Configuration
# ======================

class Config:
    CLASS_LABELS = [
        'belly_pain', 'burping', 'discomfort', 'hungry', 'tired',
        'cry', 'laugh', 'noise', 'NonSnoring', 'silence', 'Snoring'
    ]
    NUM_CLASSES = len(CLASS_LABELS)
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MIN_SAMPLES = 100
    MAX_FRAMES = 50
    N_MFCC = 10
    SAMPLE_RATE = 22050
    N_FFT = 512  # Reduced from 2048 to avoid warning
    HOP_LENGTH = 256

# ======================
# Setup Reproducibility
# ======================
def setup_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

setup_seeds()

# ======================
# Data Loading and Balancing
# ======================

def load_and_balance_data(path, label):
    print(f"\n=== Processing {os.path.basename(path)} ===")
    mfcc_data = []
    print(f"Processing category: {os.path.basename(path)}")
    
    for file in os.listdir(path):
        if file.endswith('.npy'):
            try:
                mfcc = np.load(os.path.join(path, file))
                
                # Ensure data is 2D
                if mfcc.ndim == 1:
                    mfcc = mfcc.reshape(-1, 1)
                elif mfcc.ndim > 2:
                    mfcc = mfcc.squeeze()
                
                # Standardize shape
                if mfcc.shape[0] < Config.MAX_FRAMES:
                    pad_width = ((0, Config.MAX_FRAMES - mfcc.shape[0]), (0, 0))
                    mfcc = np.pad(mfcc, pad_width, mode='constant')
                else:
                    mfcc = mfcc[:Config.MAX_FRAMES, :]
                
                if mfcc.shape[1] < Config.N_MFCC:
                    pad_width = ((0, 0), (0, Config.N_MFCC - mfcc.shape[1]))
                    mfcc = np.pad(mfcc, pad_width, mode='constant')
                elif mfcc.shape[1] > Config.N_MFCC:
                    mfcc = mfcc[:, :Config.N_MFCC]
                
                if mfcc.shape != (Config.MAX_FRAMES, Config.N_MFCC):
                    continue
                
                mfcc_data.append(mfcc)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    
    if len(mfcc_data) == 0:
        raise ValueError(f"No valid data in category {os.path.basename(path)}")
 
    print(f"Initial samples: {len(mfcc_data)}")
    iteration = 0

    # Data augmentation
    while len(mfcc_data) < Config.MIN_SAMPLES:
        iteration += 1
        print(f"Augmentation iteration {iteration}, Current samples: {len(mfcc_data)}")
        idx = np.random.randint(0, len(mfcc_data))
        original = mfcc_data[idx]
        augmented = augment_mfcc(original)
        for arr in augmented:
            if arr.shape == (Config.MAX_FRAMES, Config.N_MFCC):
                mfcc_data.append(arr)
    
    # Select samples
    mfcc_array = np.stack(mfcc_data)
    selected_indices = np.random.choice(
        len(mfcc_array),
        size=Config.MIN_SAMPLES,
        replace=len(mfcc_array) < Config.MIN_SAMPLES
    )
    return mfcc_array[selected_indices], np.full(Config.MIN_SAMPLES, label)

# ======================
# Data Augmentation
# ======================
def augment_mfcc(mfcc):
    choice = np.random.choice(['noise', 'shift', 'none'])
    augmented = []
    
    if choice == 'noise':
        new_sample = mfcc + np.random.normal(0, 0.01, mfcc.shape)
    elif choice == 'shift':
        new_sample = np.roll(mfcc, np.random.randint(-3,3), axis=0)
    else:
        new_sample = mfcc
    
    # Always return list with at least 1 sample
    return [new_sample]

def adjust_shape(mfcc, max_frames, n_mfcc):
    """Ensure MFCC has correct shape"""
    if mfcc.shape[0] < max_frames:
        pad_width = ((0, max_frames - mfcc.shape[0]), (0, 0))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:max_frames, :]
    return mfcc

# ======================
# Model Architecture
# ======================

# In your build_model() function, consider these enhancements:
def build_model():
    model = tf.keras.Sequential([
        InputLayer(shape=(Config.MAX_FRAMES, Config.N_MFCC, 1)),
        BatchNormalization(),
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        
        Conv2D(128, (3,3), activation='relu', padding='same'),
        GlobalAveragePooling2D(),
        Dropout(0.4),
        
        Dense(Config.NUM_CLASSES, activation='softmax', dtype='float32')  # Note dtype
    ])

    model.compile(
        optimizer=AdamW(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
    )
    return model



# ======================
# Main Training Function
# ======================
def main():
    base_dir = r'D:\ISDN2002\mfcc_value'
    
    # Load and balance data
    x, y = [], []
    for label, category in enumerate(Config.CLASS_LABELS):
        data_path = os.path.join(base_dir, category)
        if not os.path.exists(data_path):
            print(f"Warning: Missing {category} folder")
            continue
        
        data, labels = load_and_balance_data(data_path, label)
        x.append(data)
        y.append(labels)
    
    x = np.concatenate(x)
    y = np.concatenate(y)
    x = np.expand_dims(x, axis=-1)  # Add channel dimension

    # Split data
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42)
    
    x_test, x_special, y_test, y_special = train_test_split(
        x_remaining, y_remaining, test_size=1/3, stratify=y_remaining, random_state=42)

    # Normalization
    mean = np.mean(x_train, axis=(0,1,2))
    std = np.std(x_train, axis=(0,1,2))
    x_train = (x_train - mean) / (std + 1e-8)
    x_test = (x_test - mean) / (std + 1e-8)
    x_special = (x_special - mean) / (std + 1e-8)
    # After normalization
    cache_path = 'preprocessed_data.npz'
    if not os.path.exists(cache_path):
        np.savez(cache_path, 
                 x_train=x_train, y_train=y_train,
                 x_test=x_test, y_test=y_test,
                 x_special=x_special, y_special=y_special,
                 mean=mean, std=std)
    # Convert labels
    y_train = to_categorical(y_train, Config.NUM_CLASSES)
    y_test = to_categorical(y_test, Config.NUM_CLASSES)
    y_special = to_categorical(y_special, Config.NUM_CLASSES)

    # Build and train model
    model = build_model()
    model.summary()


    # Class weights
    class_counts = np.sum(y_train, axis=0)
    class_weights = compute_class_weight('balanced', 
                                       classes=np.arange(Config.NUM_CLASSES), 
                                       y=np.argmax(y_train, axis=1))
    class_weights = {i:w for i,w in enumerate(class_weights)}
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]    

    # Convert data to TensorFlow Dataset for better performance
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Then use datasets in model.fit()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )

    # Evaluation
    model.load_weights('best_model.keras')
    
    # Get all evaluation metrics
    test_metrics = model.evaluate(x_test, y_test, verbose=0)
    test_loss = test_metrics[0]
    test_acc = test_metrics[1]  # Assuming accuracy is the second metric
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Special data evaluation
    special_metrics = model.evaluate(x_special, y_special, verbose=0)
    special_loss = special_metrics[0]
    special_acc = special_metrics[1]
    print(f"\nSpecial Data Loss: {special_loss:.4f}")
    print(f"Special Data Accuracy: {special_acc:.4f}")

    # Save normalization stats
    np.save('mean.npy', mean)
    np.save('std.npy', std)

    # Confusion matrix
    plot_confusion_matrix(model, x_test, y_test, "Test Set Confusion Matrix")
    plot_confusion_matrix(model, x_special, y_special, "Special Data Confusion Matrix")


def plot_confusion_matrix(model, x, y, title):
    y_pred = np.argmax(model.predict(x), axis=1)
    y_true = np.argmax(y, axis=1)
    
    plt.figure(figsize=(12,10))
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True, fmt='d',
                xticklabels=Config.CLASS_LABELS,
                yticklabels=Config.CLASS_LABELS)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

if __name__ == '__main__':
    main()
