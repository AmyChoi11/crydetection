import warnings
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from awgn import awgn

# 忽略警告
warnings.simplefilter("ignore", DeprecationWarning)

# 读取音频文件
file_path = r"D:\ISDN2002\643D64AD-B711-469A-AF69-55C0D5D3E30F-1430138536-1.0-m-72-bp - 副本.wav"
framerate, waveData = wavfile.read(file_path)  # 读取音频文件

# 如果音频是双声道，waveData 的形状为 (nframes, nchannels)
# 将音频数据归一化到 [-1, 1] 范围
waveData = waveData.astype(np.float32)  # 转换为浮点数
waveData = waveData / np.max(np.abs(waveData))  # 归一化

# 添加高斯白噪声
snr = 0  # 信噪比 (dB)
waveData2 = awgn(waveData, snr, out='signal', method='vectorized', axis=0)

# 将加噪后的信号转换回 16 位整数格式
waveData2_int16 = np.int16(waveData2 * 32767)  # 16 位整数的范围是 [-32768, 32767]

# 导出加噪后的音频为 WAV 文件
output_file_path = r"D:\ISDN2002\addnoise.wav"
wavfile.write(output_file_path, framerate, waveData2_int16)

print(f"加噪后的音频已保存到: {output_file_path}")

# 绘制原始信号和加噪后的信号
plt.figure(figsize=(10, 8))

# 绘制原始信号
plt.subplot(2, 1, 1)
if waveData.ndim == 1:  # 单声道
    plt.plot(waveData)
else:  # 多声道
    plt.plot(waveData[:, 0])  # 只绘制第一个声道
plt.title('Original Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')

# 绘制加噪后的信号
plt.subplot(2, 1, 2)
if waveData2.ndim == 1:  # 单声道
    plt.plot(waveData2)
else:  # 多声道
    plt.plot(waveData2[:, 0])  # 只绘制第一个声道
plt.title('Noisy Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
