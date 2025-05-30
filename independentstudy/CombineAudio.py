import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# 讀取第一個音頻文件
f = wave.open(r"D:\ISDN2002\643D64AD-B711-469A-AF69-55C0D5D3E30F-1430138536-1.0-m-72-bp - 副本.wav", "r")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)  # 讀取音頻，字符串格式
waveData = np.frombuffer(strData, dtype=np.int16)  # 將字符串轉化為 int
waveData = waveData * 1.0 / (max(abs(waveData)))  # wave 幅值歸一化
waveData = np.reshape(waveData, [nframes, nchannels]).T  # 轉置為 (nchannels, nframes)

# 讀取第二個音頻文件
g = wave.open(r"D:\ISDN2002\noise\car_road1.wav", "r")
params2 = g.getparams()
nchannels2, sampwidth2, framerate2, nframes2 = params2[:4]
strData2 = g.readframes(nframes2)  # 讀取音頻，字符串格式
waveData2 = np.frombuffer(strData2, dtype=np.int16)  # 將字符串轉化為 int
waveData2 = waveData2 * 1.0 / (max(abs(waveData2)))  # wave 幅值歸一化
waveData2 = np.reshape(waveData2, [nframes2, nchannels2]).T  # 轉置為 (nchannels2, nframes2)

# 打印採樣率
print("f 音頻採樣率：", framerate)
print("g 音頻採樣率：", framerate2)

# 如果採樣率不一致，將 g 音頻的採樣率轉換為與 f 音頻一致
if framerate != framerate2:
    print("兩個音頻文件的採樣率不一致，進行重採樣...")
    new_nframes2 = int(nframes2 * framerate / framerate2)
    waveData2 = resample(waveData2, new_nframes2, axis=1)
    nframes2 = new_nframes2
    framerate2 = framerate
# 假設 g 音頻需要延遲 1 秒
delay_frames = int(30.0 * framerate)  # 延遲的幀數
waveData2 = np.roll(waveData2, delay_frames, axis=1)  # 延遲 g 音頻
# 處理聲道數不一致的情況
if nchannels != nchannels2:
    print("兩個音頻文件的聲道數不一致，進行處理...")
    if nchannels == 1 and nchannels2 > 1:
        # 將單聲道音頻擴展為多聲道
        waveData = np.tile(waveData, (nchannels2, 1))
        nchannels = nchannels2
    elif nchannels > 1 and nchannels2 == 1:
        # 將單聲道音頻擴展為多聲道
        waveData2 = np.tile(waveData2, (nchannels, 1))
        nchannels2 = nchannels
    else:
        # 將多聲道音頻轉為單聲道
        waveData = np.mean(waveData, axis=0, keepdims=True)
        waveData2 = np.mean(waveData2, axis=0, keepdims=True)
        nchannels = 1
        nchannels2 = 1

# 確定合併後的音頻長度（與較短的音頻一致）
merged_length = min(nframes, nframes2)

# 合併音頻數據
new = np.zeros(shape=(nchannels, merged_length))
for i in range(nchannels):
    # 截斷較長的音頻
    rwaveData = waveData[i][:merged_length]
    rwaveData2 = waveData2[i][:merged_length]
    
    # 加權合併兩個音頻數據（避免幅值超出範圍）
    weight_f = 0.4  # f 音頻的加權系數
    weight_g = 0.6  # g 音頻的加權系數
    new_waveData = (rwaveData * weight_f) + (rwaveData2 * weight_g)
    
    # 檢查合併後的音頻數據幅值範圍
    print(f"Channel {i + 1} 合併後的音頻數據幅值範圍：", np.min(new_waveData), np.max(new_waveData))
    
    new_waveData = new_waveData / np.max(np.abs(new_waveData))  # 歸一化到 [-1, 1]
    new[i] = new_waveData

# 將合併後的音頻數據轉換為 int16 格式
new_int16 = (new * 32767).astype(np.int16)  # 縮放到 int16 範圍

# 導出合併後的音頻數據到 WAV 文件
output_file = "merged_audio_car.wav"
with wave.open(output_file, 'wb') as out:
    out.setnchannels(nchannels)  # 設置聲道數
    out.setsampwidth(2)  # 設置採樣寬度（2 字節，即 16 位）
    out.setframerate(framerate)  # 設置採樣率
    out.writeframes(new_int16.T.tobytes())  # 寫入音頻數據

print(f"合併後的音頻已導出到 {output_file}")

f.close()
g.close()

# 繪製原始音頻波形
plt.figure(1)
for i in range(nchannels):
    plt.subplot(2, 2, i + 1)
    plt.plot(waveData[i])
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    plt.title(f'Channel {i + 1} - Original')

# 繪製合併後的音頻波形
plt.figure(2)
for i in range(nchannels):
    plt.subplot(2, 2, i + 1)
    plt.plot(new[i])
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    plt.title(f'Channel {i + 1} - Merged')

plt.show()
