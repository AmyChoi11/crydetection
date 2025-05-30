import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
# 讀取音訊文件
sample_rate, audio = wavfile.read(r"D:\ISDN2002\testing\input\Audio.wav")

# 將立體聲轉為單聲道
if len(audio.shape) == 2:
    audio = np.mean(audio, axis=1)

# 分析雜音
noise = audio[:int(0.1 * sample_rate)]  # 取前 0.5 秒當雜音
noise_spectrum = np.abs(np.fft.fft(noise))  # 用 FFT 分析雜音的頻率
noise_spectrum = gaussian_filter(noise_spectrum, sigma=1)  # 平滑處理

# 分段處理（帶重疊）
frame_size = int(0.02 * sample_rate)  # 20ms 一段
overlap = int(0.7 * frame_size)  # 50% 重疊
frames = [audio[i:i+frame_size] for i in range(0, len(audio) - frame_size, frame_size - overlap)]
spectrums = [np.fft.fft(frame) for frame in frames]

# 減掉雜音
clean_spectrums = []
for spectrum in spectrums:
    magnitude = np.abs(spectrum) - noise_spectrum[:len(spectrum)]
    magnitude = np.maximum(magnitude, 1e-6)  # 避免完全截斷
    clean_spectrum = magnitude * np.exp(1j * np.angle(spectrum))  # 保留相位
    clean_spectrums.append(clean_spectrum)

# 合併聲音（重疊相加）
clean_frames = [np.fft.ifft(spectrum).real for spectrum in clean_spectrums]
clean_audio = np.zeros(len(audio))
for i, frame in enumerate(clean_frames):
    start = i * (frame_size - overlap)
    clean_audio[start:start+frame_size] += frame

# 歸一化並保存
clean_audio = clean_audio / np.max(np.abs(clean_audio))  # 歸一化到 [-1, 1]
clean_audio = (clean_audio * 32767).astype(np.int16)  # 縮放到 int16 範圍
wavfile.write("Clean.wav", sample_rate, clean_audio)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(audio)
plt.title("Original Audio")
plt.subplot(2, 1, 2)
plt.plot(clean_audio)
plt.title("Denoised Audio")
plt.show()

