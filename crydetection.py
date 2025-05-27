from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import scipy.stats
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, InputLayer, Add, Input
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import librosa
import random
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

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
        'Unwell', 'Sleeping', 'Cry',
        'Laugh', 'Tired', 'Silence'
    ]
    NUM_CLASSES = len(CLASS_LABELS)
    BATCH_SIZE = 64
    EPOCHS = 120  # Increased epochs with better early stopping
    LEARNING_RATE = 0.001
    MIN_SAMPLES = 100
    MAX_FRAMES = 50
    N_MFCC = 13  # Standard MFCC coefficient count
    SAMPLE_RATE = 22050
    N_FFT = 512
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
    
    # Use standard coefficient count
    fixed_n_mfcc = 13
    
    for file in os.listdir(path):
        if file.endswith('.npy'):
            try:
                mfcc = np.load(os.path.join(path, file))           
                # Process the entire MFCC regardless of length
                if mfcc.shape[0] > 0:
                    # Handle different coefficient counts
                    if mfcc.shape[1] != fixed_n_mfcc:
                        # If fewer coefficients, pad with zeros
                        if mfcc.shape[1] < fixed_n_mfcc:
                            pad_width = ((0, 0), (0, fixed_n_mfcc - mfcc.shape[1]))
                            mfcc = np.pad(mfcc, pad_width, mode='constant')
                        # If more coefficients, truncate
                        else:
                            mfcc = mfcc[:, :fixed_n_mfcc]
                    
                    # Statistical features
                    mean_features = np.mean(mfcc, axis=0)  # Central tendency
                    std_features = np.std(mfcc, axis=0)    # Variability
                    max_features = np.max(mfcc, axis=0)    # Peak values
                    
                    # If we have enough frames for temporal analysis
                    if mfcc.shape[0] > 1:
                        # Temporal variability (how quickly features change)
                        delta_features = np.std(np.diff(mfcc, axis=0), axis=0)
                        
                        # Entropy (randomness measure)
                        # Add small constant to avoid log(0)
                        entropy_features = scipy.stats.entropy(np.abs(mfcc) + 1e-10, axis=0)
                        
                        # NEW: Zero-crossing rate - helps distinguish speech vs. noise
                        # Approximate from MFCC
                        zcr_features = np.mean(np.abs(np.diff(np.sign(mfcc), axis=0)), axis=0) / 2
                        
                        # NEW: Quartile features - help distinguish sound distributions
                        q1_features = np.percentile(mfcc, 25, axis=0)
                        q3_features = np.percentile(mfcc, 75, axis=0)
                        
                        # NEW: Envelope features - help with detecting cries vs other sounds
                        # Calculate temporal envelope 
                        env = np.max(np.abs(mfcc), axis=1)
                        env_std = np.std(env) * np.ones(fixed_n_mfcc)
                        env_rate = np.mean(np.abs(np.diff(env))) * np.ones(fixed_n_mfcc)
                        
                        # Combine all features
                        combined_features = np.concatenate([
                            mean_features, std_features, max_features, 
                            delta_features, entropy_features, 
                            zcr_features, q1_features, q3_features,
                            env_std, env_rate
                        ])
                    else:
                        # Fallback if very short audio - fill with zeros
                        extra_features = np.zeros_like(mean_features)
                        combined_features = np.concatenate([
                            mean_features, std_features, max_features, 
                            extra_features, extra_features,
                            extra_features, extra_features, extra_features,
                            extra_features, extra_features
                        ])
                    
                    mfcc_data.append(combined_features)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    
    if len(mfcc_data) == 0:
        raise ValueError(f"No valid data in category {os.path.basename(path)}")
 
    print(f"Initial samples: {len(mfcc_data)}")
    iteration = 0

    # Improved data augmentation with smarter stopping
    while len(mfcc_data) < Config.MIN_SAMPLES and iteration < 50:  # Limit iterations to avoid infinite loop
        iteration += 1
        if iteration % 5 == 0:
            print(f"Augmentation iteration {iteration}, Current samples: {len(mfcc_data)}")
            
        # Generate more diverse augmentations
        idx = np.random.randint(0, len(mfcc_data))
        original = mfcc_data[idx]
        augmented = enhanced_augment_mfcc(original)
        
        for arr in augmented:
            if arr.shape[0] == original.shape[0]:  # Match feature size
                mfcc_data.append(arr)
                if len(mfcc_data) >= Config.MIN_SAMPLES:
                    break
    
    # Select samples with limited replacement
    mfcc_array = np.stack(mfcc_data)
    if len(mfcc_array) >= Config.MIN_SAMPLES:
        selected_indices = np.random.choice(
            len(mfcc_array),
            size=Config.MIN_SAMPLES,
            replace=False
        )
    else:
        # Only replace what's needed
        num_actual = len(mfcc_array)
        num_replacement = Config.MIN_SAMPLES - num_actual
        
        # All real samples + some repeated
        indices_actual = np.arange(num_actual)
        indices_replacement = np.random.choice(num_actual, size=num_replacement, replace=True)
        selected_indices = np.concatenate([indices_actual, indices_replacement])
        
    return mfcc_array[selected_indices], np.full(Config.MIN_SAMPLES, label)

# ======================
# Enhanced Data Augmentation
# ======================
def enhanced_augment_mfcc(features):
    """Enhanced augmentation for statistical features"""
    results = []
    feature_size = len(features)
    n_mfcc = Config.N_MFCC  # Each feature contributes multiple statistics
    num_feature_types = feature_size // n_mfcc  # How many different feature types we have
    
    # 1. Original method with stronger noise for robustness
    noise_scale = np.random.uniform(0.001, 0.02)
    results.append(features + np.random.normal(0, noise_scale, features.shape))
    
    # 2. Scale features slightly (simulate volume/intensity changes)
    scale = np.random.uniform(0.9, 1.1)
    results.append(features * scale)
    
    # 3. Emphasize different feature regions
    for i in range(num_feature_types):  # Different regions of features
        boost = np.ones(feature_size)
        start_idx = i * n_mfcc
        end_idx = (i + 1) * n_mfcc
        boost[start_idx:end_idx] = np.random.uniform(0.8, 1.2)
        results.append(features * boost)
    
    # 4. Mix adjacent features slightly (simulate slight time shifts)
    idx = np.random.randint(0, len(features) // num_feature_types)
    mixed = features.copy()
    mix_factor = np.random.uniform(0.1, 0.3)
    mixed[idx] = mixed[idx] * (1-mix_factor) + mixed[(idx+1) % n_mfcc] * mix_factor
    results.append(mixed)
    
    # 5. NEW: Targeted feature boosting for specific categories
    # This helps improve discrimination between commonly confused categories
    category_specific = features.copy()
    # Enhance high-frequency content (helps distinguish noise from speech)
    for i in range(3):  # Focus on mean, std, max
        start_idx = i * n_mfcc + n_mfcc//2  # Boost higher mfcc coefficients
        end_idx = (i+1) * n_mfcc
        category_specific[start_idx:end_idx] *= np.random.uniform(1.05, 1.15)
    results.append(category_specific)
    
    # 6. NEW: Feature masking (helps with robustness)
    masked = features.copy()
    mask_idx = np.random.randint(0, num_feature_types)
    start_idx = mask_idx * n_mfcc
    end_idx = (mask_idx + 1) * n_mfcc
    masked[start_idx:end_idx] *= np.random.uniform(0.7, 0.9)  # Partially mask
    results.append(masked)
    
    return results

# ======================
# Mixup Model (Fixed for robustness)
# Replace your MixupModel class with this more robust implementation:

class MixupModel(tf.keras.Model):
    def __init__(self, base_model, alpha=0.2):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha
        
    def compile(self, **kwargs):
        super().compile(**kwargs)
        # Ensure base model has same metrics for proper sharing
        self.base_model.compile(**kwargs)
        
    def train_step(self, data):
        # Simplify data unpacking - rely on TensorFlow's standard format
        if isinstance(data, tuple):
            x, y = data
        else:
            raise ValueError(f"Expected tuple data, got {type(data)}")
            
        batch_size = tf.shape(x)[0]
        
        # Generate mixing parameters
        alpha = tf.constant(self.alpha, dtype=tf.float32)
        weight = tf.random.beta(alpha, alpha, shape=[batch_size, 1])
        
        # Create mixed samples
        indices = tf.random.shuffle(tf.range(batch_size))
        x_mixed = weight * x + (1 - weight) * tf.gather(x, indices)
        y_mixed = weight * y + (1 - weight) * tf.gather(y, indices)
        
        # Train on mixed samples
        with tf.GradientTape() as tape:
            y_pred = self.base_model(x_mixed, training=True)
            loss = self.compiled_loss(y_mixed, y_pred)
        
        # Compute & apply gradients
        trainable_vars = self.base_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics - use pure samples for metric evaluation
        y_pred_pure = self.base_model(x, training=False)
        self.compiled_metrics.update_state(y, y_pred_pure)
        
        # Return metrics dictionary
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results
    
    def call(self, inputs):
        return self.base_model(inputs)

# ======================
# Model Architecture
# ======================
def build_model():
    # Get feature size dynamically (since we've added more features)
    feature_size = Config.N_MFCC * 10  # Now using 10 statistics per coefficient
    print(f"Building model with input size: {feature_size}")
    
    # Using Functional API for more flexibility
    inputs = Input(shape=(feature_size,))
    x = BatchNormalization()(inputs)
    
    # First dense block
    x1 = Dense(256, activation='relu', kernel_regularizer=l2(2e-5))(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    # Second dense block with residual connection
    x2 = Dense(128, activation='relu', kernel_regularizer=l2(2e-5))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Residual connection - transform input to match dimensions with x2
    x_res = Dense(128, activation='linear')(x)
    x2_combined = Add()([x2, x_res])  # Add residual connection
    
    # Third dense block
    x3 = Dense(64, activation='relu', kernel_regularizer=l2(2e-5))(x2_combined)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.2)(x3)
    
    # Output layer
    outputs = Dense(Config.NUM_CLASSES, activation='softmax', dtype='float32')(x3)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Use a learning rate scheduler instead of a custom callback
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=Config.LEARNING_RATE,
        decay_steps=Config.EPOCHS,
        alpha=0.1  # Minimum learning rate as a fraction of initial rate
    )
    
    # Use AdamW with weight decay for better generalization
    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=2e-5,
        beta_1=0.9,
        beta_2=0.99  # Slightly higher beta2 for more stable updates
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    return model

# ======================
# Main Training Function
# ======================
def main():
    base_dir = r'D:\ISDN2002\mfcc_value'
    # Check for cached preprocessed data
    cache_path = 'preprocessed_data.npz'
    use_cache = False
    
    if os.path.exists(cache_path):
        try:
            print("Loading cached preprocessed data...")
            cached = np.load(cache_path)
            x_train = cached['x_train']
            y_train = cached['y_train']
            x_test = cached['x_test']
            y_test = cached['y_test']
            x_special = cached['x_special']
            y_special = cached['y_special']
            mean = cached['mean']
            std = cached['std']
            
            # Verify if dimensions match the current model (10 stats vs 5)
            expected_feature_size = Config.N_MFCC * 10
            if x_train.shape[1] == expected_feature_size:
                use_cache = True
            else:
                print(f"Cache has wrong feature size ({x_train.shape[1]})! Regenerating data...")
                # Try to delete the cache file instead of renaming it
                try:
                    # Close the numpy file first to release the handle
                    cached.close()
                    # Wait a moment to ensure the file is closed
                    import time
                    time.sleep(1)
                    # Try to delete the file
                    os.remove(cache_path)
                    print("Removed outdated cache file.")
                except Exception as e:
                    print(f"Could not remove cache file: {e}")
                    print("Will create new cache when processing is complete.")
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    if not use_cache:
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
    
        # Split data
        x_train, x_remaining, y_train, y_remaining = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=42)
        
        x_test, x_special, y_test, y_special = train_test_split(
            x_remaining, y_remaining, test_size=1/3, stratify=y_remaining, random_state=42)
    
        # Normalization
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean) / (std + 1e-8)
        x_test = (x_test - mean) / (std + 1e-8)
        x_special = (x_special - mean) / (std + 1e-8)
        
        # Save processed data
        np.savez(cache_path, 
                 x_train=x_train, y_train=y_train,
                 x_test=x_test, y_test=y_test,
                 x_special=x_special, y_special=y_special,
                 mean=mean, std=std)

    # Convert labels
    y_train_cat = to_categorical(y_train, Config.NUM_CLASSES)
    y_test_cat = to_categorical(y_test, Config.NUM_CLASSES)
    y_special_cat = to_categorical(y_special, Config.NUM_CLASSES)

    # Build and train model
    model = build_model()
    model.summary()

    # Class weights to handle imbalance
    class_weights = compute_class_weight('balanced', 
                                       classes=np.unique(y_train), 
                                       y=y_train)
    class_weight_dict = {i:w for i,w in enumerate(class_weights)}
    
    # Enhance weights for problem categories based on confusion matrix
    category_difficulty = {
        "Sleeping": 1.5,  # Often confused with Noise
        "Tired": 1.5,     # Often confused with Unwell
        #"Noise": 1.3,     # Scattered predictions
        "Laugh": 1.2,     # Often confused with Cry
    }
    
    # Apply difficulty weights to class weights
    for idx, category in enumerate(Config.CLASS_LABELS):
        if category in category_difficulty:
            class_weight_dict[idx] *= category_difficulty[category]
    
    # Enhanced callbacks - Removed custom CyclicLR and using built-in
    callbacks = [
        # Better early stopping
        EarlyStopping(
            monitor='val_accuracy', 
            patience=20,  # More patience for improved convergence
            restore_best_weights=True, 
            min_delta=0.005,
            verbose=1
        ),
        # Save checkpoints
        ModelCheckpoint(
            'newest_model.keras', 
            save_best_only=True, 
            monitor='val_accuracy',
            verbose=1
        ),
        # Use built-in ReduceLROnPlateau instead of custom CyclicLR
        
    ]    

    # Convert data to TensorFlow Dataset for better performance
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat))
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    
    # Standard dataset without mixup
    train_dataset_batched = train_dataset.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat))
    val_dataset = val_dataset.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Phase 1 training with standard model (skip mixup)
    print("\n=== Phase 1: Training ===")
    history1 = model.fit(
        train_dataset_batched,
        validation_data=val_dataset,
        epochs=Config.EPOCHS // 2,  
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Load best model so far
    model.load_weights('newest_model.keras')

    # Phase 2 with regular model (fine-tuning)
    print("\n=== Phase 2: Fine-tuning ===")
    history2 = model.fit(
        train_dataset_batched,
        validation_data=val_dataset,
        epochs=Config.EPOCHS // 2,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Combine histories for plotting
    combined_history = {}
    for k in history1.history.keys():
        if k in history2.history:
            combined_history[k] = history1.history[k] + history2.history[k]
    
    # Create a History object for plotting
    history = type('obj', (object,), {'history': combined_history})
    
    # Evaluation
    model.load_weights('newest_model.keras')
    
    # Get all evaluation metrics
    test_metrics = model.evaluate(x_test, y_test_cat, verbose=0)
    test_loss = test_metrics[0]
    test_acc = test_metrics[1]  # Assuming accuracy is the second metric
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Special data evaluation
    special_metrics = model.evaluate(x_special, y_special_cat, verbose=0)
    special_loss = special_metrics[0]
    special_acc = special_metrics[1]
    print(f"\nSpecial Data Loss: {special_loss:.4f}")
    print(f"Special Data Accuracy: {special_acc:.4f}")

    # Save normalization stats
    np.save('mean.npy', mean)
    np.save('std.npy', std)
    
    # Plot training history
    plot_training_history(history)

    # Confusion matrices
    plot_confusion_matrix(model, x_test, y_test_cat, "Test Set Confusion Matrix")
    plot_confusion_matrix(model, x_special, y_special_cat, "Special Data Confusion Matrix")

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(model, x, y, title):
    """Plot and save confusion matrix"""
    y_pred = np.argmax(model.predict(x), axis=1)
    y_true = np.argmax(y, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    
    # Plot with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASS_LABELS,
                yticklabels=Config.CLASS_LABELS)
    plt.title(f"{title} (Accuracy: {np.sum(np.diag(cm))/np.sum(cm):.2%})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()