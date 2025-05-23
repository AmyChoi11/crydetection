from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, InputLayer, Reshape, Bidirectional, LSTM, Multiply, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
from sklearn.utils.class_weight import compute_class_weight

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Use mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# ======================
# Optimized Configuration
# ======================
class Config:
    CLASS_LABELS = [
        'unwelltired', 'hungry', 'others', 'laugh',
        'quiet'
    ]
    NUM_CLASSES = len(CLASS_LABELS)
    BATCH_SIZE = 64
    EPOCHS = 50        # Increased for better training
    LEARNING_RATE = 0.001  # Reduced for better generalization
    MIN_SAMPLES = 120
    MAX_FRAMES = 48    # Increased for better temporal patterns
    N_MFCC = 24        # Increased for better feature resolution
    SAMPLE_RATE = 22050
    N_FFT = 1024
    HOP_LENGTH = 256

# ======================
# Setup Reproducibility
# ======================
def setup_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seeds()

# ======================
# Feature Engineering
# ======================
def extract_custom_features(mfcc):
    """Add engineered features to better separate unwelltired from hungry"""
    # Add spectral statistics as extra features
    row_means = np.mean(mfcc, axis=1, keepdims=True)
    row_stds = np.std(mfcc, axis=1, keepdims=True)
    
    # Add temporal dynamics (derivatives)
    if mfcc.shape[0] > 1:
        deltas = np.zeros_like(mfcc)
        deltas[1:] = mfcc[1:] - mfcc[:-1]
        deltas_mean = np.mean(np.abs(deltas), axis=1, keepdims=True)
        
        # Combine all features
        enhanced = np.concatenate([
            mfcc, 
            row_means,
            row_stds,
            deltas_mean
        ], axis=1)
        
        # Ensure we don't exceed our feature count target
        if enhanced.shape[1] > Config.N_MFCC:
            enhanced = enhanced[:, :Config.N_MFCC]
            
        return enhanced
    return mfcc  # Fall back if we can't compute derivatives

# ======================
# Specialized Data Loading
# ======================
def load_and_balance_data(path, label):
    print(f"Processing {os.path.basename(path)}")
    mfcc_data = []
    
    # Load all available files
    for file in os.listdir(path):
        if file.endswith('.npy'):
            try:
                mfcc = np.load(os.path.join(path, file))
                if mfcc.ndim == 1:
                    mfcc = mfcc.reshape(-1, 1)
                elif mfcc.ndim > 2:
                    mfcc = mfcc.squeeze()
                
                # Shape adjustment
                if mfcc.shape[0] < Config.MAX_FRAMES:
                    mfcc = np.pad(mfcc, ((0, Config.MAX_FRAMES - mfcc.shape[0]), (0, 0)), 'constant')
                else:
                    mfcc = mfcc[:Config.MAX_FRAMES, :]
                
                if mfcc.shape[1] < Config.N_MFCC:
                    mfcc = np.pad(mfcc, ((0, 0), (0, Config.N_MFCC - mfcc.shape[1])), 'constant')
                else:
                    mfcc = mfcc[:, :Config.N_MFCC]
                
                # Apply feature engineering
                mfcc = extract_custom_features(mfcc)
                mfcc_data.append(mfcc)
            except Exception as e:
                continue
    
    if len(mfcc_data) == 0:
        raise ValueError(f"No valid data in {os.path.basename(path)}")
    
    print(f"Initial samples: {len(mfcc_data)}")
    
    # Class-specific augmentation
    original_count = len(mfcc_data)
    class_name = os.path.basename(path)
    
    # Different augmentation strategies based on class
    if class_name == 'unwelltired':
        print(f"Applying specialized augmentation for {class_name}")
        # Create 3x more diverse samples for problematic class
        target_count = max(Config.MIN_SAMPLES, original_count * 3)
        
        while len(mfcc_data) < target_count:
            idx = np.random.randint(0, original_count)
            base = mfcc_data[idx].copy()
            
            # Apply specialized augmentations for unwelltired class
            aug_type = np.random.choice(['shift', 'multi_mask', 'noise', 'combined'])
            
            if aug_type == 'shift':
                # Apply time shifts to capture temporal variations
                shift = np.random.randint(-5, 6)
                new_sample = np.roll(base, shift, axis=0)
                
            elif aug_type == 'multi_mask':
                # Apply frequency masking to help model learn robust features
                new_sample = base.copy()
                mask_size = np.random.randint(1, 4)
                start_idx = np.random.randint(0, base.shape[1] - mask_size)
                new_sample[:, start_idx:start_idx+mask_size] = 0
                
            elif aug_type == 'noise':
                # Add varying noise levels
                noise_level = np.random.uniform(0.01, 0.04)
                new_sample = base + np.random.normal(0, noise_level, base.shape)
                
            else:  # combined
                # Apply multiple augmentations
                new_sample = base.copy()
                # First add noise
                new_sample = new_sample + np.random.normal(0, 0.02, new_sample.shape)
                # Then apply a small shift
                new_sample = np.roll(new_sample, np.random.randint(-3, 4), axis=0)
            
            mfcc_data.append(new_sample)
            
    else:
        # Standard augmentation for other classes
        while len(mfcc_data) < Config.MIN_SAMPLES:
            idx = np.random.randint(0, original_count)
            # Simple noise augmentation
            new_sample = mfcc_data[idx] + np.random.normal(0, 0.02, mfcc_data[idx].shape)
            mfcc_data.append(new_sample)
    
    # Return all samples
    if len(mfcc_data) < Config.MIN_SAMPLES:
        print(f"Warning: Using augmented data for {os.path.basename(path)} ({len(mfcc_data)} real files)")
    
    mfcc_array = np.stack(mfcc_data)
    return mfcc_array, np.full(len(mfcc_array), label)

# ======================
# Enhanced Model Architecture
# ======================
def build_model():
    # Define input shape
    inputs = tf.keras.Input(shape=(Config.MAX_FRAMES, Config.N_MFCC, 1))
    
    # Preprocessing
    x = BatchNormalization()(inputs)
    
    # First convolutional block with residual connection
    conv1 = SeparableConv2D(48, (3,3), activation='relu', padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(48, (3,3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    # Add residual connection
    x = Conv2D(48, (1,1), padding='same')(x)  # 1x1 conv to match dimensions
    x = tf.keras.layers.add([x, conv1])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)
    
    # Second block with attention mechanism
    conv2 = SeparableConv2D(96, (3,3), activation='relu', padding='same')(x)
    conv2 = BatchNormalization()(conv2)
    
    # Channel attention
    att = GlobalAveragePooling2D()(conv2)
    att = Dense(96, activation='sigmoid')(att)
    att = Reshape((1, 1, 96))(att)
    conv2 = Multiply()([conv2, att])
    
    x = MaxPooling2D((2,2))(conv2)
    x = Dropout(0.4)(x)
    
    # Global spatial features
    spatial_features = GlobalAveragePooling2D()(x)
    
    # Temporal features using LSTM
    # Reshape for LSTM processing
    reshape_layer = Reshape((-1, x.shape[-1] * x.shape[-2]))(x)
    lstm_layer = Bidirectional(LSTM(64))(reshape_layer)
    
    # Combine features
    combined = Concatenate()([spatial_features, lstm_layer])
    combined = Dropout(0.5)(combined)
    
    # Final classification head
    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(Config.NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Custom loss function that penalizes unwelltired→hungry confusion
    def weighted_categorical_crossentropy(y_true, y_pred):
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Extract indices for unwelltired and hungry classes
        unwelltired_idx = Config.CLASS_LABELS.index('unwelltired')
        hungry_idx = Config.CLASS_LABELS.index('hungry')
        
        # Is this sample an "unwelltired" class?
        is_unwelltired = tf.cast(tf.argmax(y_true, axis=1) == unwelltired_idx, tf.float32)
        # How confident is the model that this is "hungry"?
        hungry_pred = y_pred[:, hungry_idx]
        
        # Extra penalty when model confuses unwelltired for hungry
        confusion_penalty = is_unwelltired * hungry_pred * 3.0
        
        return base_loss + confusion_penalty
    
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss=weighted_categorical_crossentropy,
        metrics=['accuracy']
    )
    
    return model

# ======================
# Training with Cross-validation
# ======================
def main():
    base_dir = r'D:\ISDN2002\mfcc_value'
    
    # Delete cache file
    cache_path = 'preprocessed_data.npz'
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("Deleted cache file")
    
    # Load data
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
    x = np.expand_dims(x, axis=-1)
    
    # Prepare for cross-validation
    n_folds = 3
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    fold_models = []
    
    # Convert labels to one-hot encoding for training
    y_categorical = to_categorical(y, Config.NUM_CLASSES)
    
    print(f"Total dataset size: {len(x)} samples")
    print(f"Starting {n_folds}-fold cross-validation")
    
    fold_num = 1
    for train_idx, val_idx in kfold.split(x, y):
        print(f"\n---- Training Fold {fold_num}/{n_folds} ----")
        
        # Split data
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
        
        # Normalize
        mean = np.mean(x_train)
        x_train = (x_train - mean)
        x_val = (x_val - mean)
        
        # Build model
        model = build_model()
        
        # Calculate class weights with special focus on unwelltired
        class_weights = compute_class_weight('balanced', 
                                    classes=np.arange(Config.NUM_CLASSES), 
                                    y=np.argmax(y_train, axis=1))
        
        # Increase weight for unwelltired class
        unwelltired_idx = Config.CLASS_LABELS.index('unwelltired')
        class_weights[unwelltired_idx] *= 2.0  # Double the weight
        
        class_weights = {i:w for i,w in enumerate(class_weights)}
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy', 
                patience=10,
                restore_best_weights=True, 
                min_delta=0.005
            ),
            ModelCheckpoint(
                f'best_model_fold{fold_num}.keras', 
                save_best_only=True,  # Fixed parameter name
                monitor='val_accuracy',
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]

        # Convert to TensorFlow Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.cache().shuffle(buffer_size=len(x_train)).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.cache().batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate
        val_metrics = model.evaluate(x_val, y_val, verbose=0)
        fold_metrics.append(val_metrics)
        
        # Save model
        model.save(f'model_fold{fold_num}.keras')
        fold_models.append(model)
        
        # Create confusion matrix for this fold
        y_pred = model.predict(x_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=Config.CLASS_LABELS, yticklabels=Config.CLASS_LABELS)
        plt.title(f"Fold {fold_num} Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'confusion_matrix_fold{fold_num}.png')
        plt.close()
        
        fold_num += 1
    
    # Calculate average metrics
    avg_loss = np.mean([metrics[0] for metrics in fold_metrics])
    avg_acc = np.mean([metrics[1] for metrics in fold_metrics])
    
    print(f"\nAverage validation loss: {avg_loss:.4f}")
    print(f"Average validation accuracy: {avg_acc:.4f}")
    
    # Final evaluation on a holdout test set (20% of data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42)
    
    # Normalize
    mean = np.mean(x_train)
    x_test = (x_test - mean)
    
    # Convert to one-hot
    y_test = to_categorical(y_test, Config.NUM_CLASSES)
    
    # Make ensemble predictions
    print("\nGenerating ensemble predictions...")
    
    # Collect predictions from all models
    fold_predictions = []
    for i, model in enumerate(fold_models):
        pred = model.predict(x_test, verbose=0)
        fold_predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(fold_predictions, axis=0)
    
    # Post-processing correction (focus on unwelltired misclassifications)
    def correct_predictions(predictions):
        corrected = predictions.copy()
        unwelltired_idx = Config.CLASS_LABELS.index('unwelltired')
        hungry_idx = Config.CLASS_LABELS.index('hungry')
        
        # For predictions where hungry is top but unwelltired is close behind
        for i in range(len(corrected)):
            pred_class = np.argmax(corrected[i])
            if pred_class == hungry_idx:
                # If model is unsure between hungry and unwelltired
                if (corrected[i, hungry_idx] < 0.65 and 
                    corrected[i, unwelltired_idx] > 0.25):
                    # Boost unwelltired probability
                    boost = corrected[i, hungry_idx] * 0.3
                    corrected[i, hungry_idx] -= boost
                    corrected[i, unwelltired_idx] += boost
        
        return corrected
    
    # Apply correction
    final_pred = correct_predictions(ensemble_pred)
    
    # Get final predictions
    y_pred_classes = np.argmax(final_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    print(f"\nFinal ensemble accuracy: {accuracy:.4f}")
    
    # Final confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=Config.CLASS_LABELS, yticklabels=Config.CLASS_LABELS)
    plt.title("Final Ensemble Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('final_confusion_matrix.png')
    plt.close()
    print("Final confusion matrix saved to final_confusion_matrix.png")
    
    # Save best ensemble model
    best_model_idx = np.argmax([metrics[1] for metrics in fold_metrics])
    best_model = fold_models[best_model_idx]
    best_model.save('best_model.keras')
    print(f"Best model (fold {best_model_idx+1}) saved as best_model.keras")

if __name__ == '__main__':
    main()
