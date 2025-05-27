import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.layers import InputLayer, BatchNormalization, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense

# ------------ 确定性设置 ------------
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# ------------ 配置类 ------------
class Config:
    CLASS_LABELS = ['Unwell', 'Sleeping', 'Cry', 'Laugh', 'Tired', 'Silence']
    NUM_CLASSES = len(CLASS_LABELS)
    MAX_FRAMES = 50  # 必须与训练时相同！
    N_MFCC = 10      # 必须与训练时相同！
    SAMPLE_RATE = 22050  # 采样率
    HOP_LENGTH = 256  # 跳跃长度
    N_FFT = 512      # FFT窗口大小

# ------------ 模型架构 (与训练一致) ------------
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
        
        Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    model.load_weights('newest_model.keras')  # 确保使用正确权重文件
    return model

# ------------ 数据预处理 ------------
def load_and_preprocess(file_path):
    # 使用librosa加载WAV文件
    print(f"Loading audio file: {file_path}")
    y, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
    
    # 提取MFCC特征
    print(f"Extracting MFCC features, audio shape: {y.shape}")
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=Config.SAMPLE_RATE,
        n_mfcc=Config.N_MFCC,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH
    )
    
    # MFCC转置以匹配模型输入格式 (frames, features)
    mfcc = mfcc.T
    print(f"MFCC shape after extraction: {mfcc.shape}")
    
    # 添加批次维度
    mfcc = np.expand_dims(mfcc, axis=0)
    
    # 时间轴对齐
    if mfcc.shape[1] < Config.MAX_FRAMES:
        pad_width = ((0,0), (0, Config.MAX_FRAMES - mfcc.shape[1]), (0,0))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:, :Config.MAX_FRAMES, :]
    
    # 添加通道维度
    mfcc = np.expand_dims(mfcc, axis=-1)
    print(f"MFCC final shape: {mfcc.shape}")
    
    # 标准化
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    return (mfcc - mean) / (std + 1e-8)

# ------------ 执行预测 ------------
if __name__ == "__main__":
    # 预处理
    data = load_and_preprocess(r"D:\ISDN2002\archive\belly_pain\69BDA5D6-0276-4462-9BF7-951799563728-1436936185-1.1-m-26-bp.wav")
    
    # 加载模型
    model = build_model()
    
    # 预测
    proba = model.predict(data, verbose=0)[0]
    label_idx = np.argmax(proba)
    
    # 输出结果
    print(f"预测结果: {Config.CLASS_LABELS[label_idx]}")
    print("各类别概率:")
    for cls, p in zip(Config.CLASS_LABELS, np.round(proba, 4)):
        print(f"  {cls}: {p:.4f}")