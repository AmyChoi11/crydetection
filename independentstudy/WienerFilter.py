import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# 读取音频文件
sample_rate, audio = wavfile.read(r"D:\ISDN2002\merged_audio_whitenoise.wav")

# 将立体声转为单声道
if len(audio.shape) == 2:
    audio = np.mean(audio, axis=1)

# 动态噪声估计
def estimate_noise(audio, sample_rate, noise_duration=0.5):
    noise = audio[:int(noise_duration * sample_rate)]
    return np.abs(np.fft.fft(noise))**2

# SNR计算
def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

noise_spectrum = estimate_noise(audio, sample_rate)
snr = calculate_snr(audio, noise_spectrum)
print("输入音频的 SNR：", snr)

# 分段处理
frame_size = int(0.02 * sample_rate)
overlap = int(0.7 * frame_size)
frames = [audio[i:i+frame_size] for i in range(0, len(audio) - frame_size, frame_size - overlap)]
spectrums = [np.fft.fft(frame) for frame in frames]

# Wiener 滤波
alpha = 1.0  # 尝试不同的 alpha 值
clean_spectrums = []
for spectrum in spectrums:
    signal_power = np.abs(spectrum)**2
    wiener = signal_power / (signal_power + alpha * noise_spectrum[:len(spectrum)])
    clean_spectrum = spectrum * wiener
    clean_spectrums.append(clean_spectrum)

# 合并声音（使用窗函数）
clean_audio = np.zeros(len(audio))
window = np.hanning(frame_size)
for i, frame in enumerate(clean_spectrums):
    start = i * (frame_size - overlap)
    clean_audio[start:start+frame_size] += np.fft.ifft(frame).real * window

# 归一化并保存
clean_audio = clean_audio / np.max(np.abs(clean_audio))
clean_audio = (clean_audio * 32767).astype(np.int16)
wavfile.write("wf_whitenoise.wav", sample_rate, clean_audio)

# 绘制波形
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(audio)
plt.title("Original Audio")
plt.subplot(2, 1, 2)
plt.plot(clean_audio)
plt.title("Denoised Audio")
plt.show()
