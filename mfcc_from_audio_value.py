import os
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
import scipy.fftpack

# ================== 参数配置 ==================
input_dir = r"D:\ISDN2002\testing\input"  # 输入音频文件夹路径
output_dir = r"D:\ISDN2002\testing\output"          # 输出MFCC值保存路径
supported_formats = ['.wav', '.m4a', '.ogg']           # 支持的音频格式
# =============================================

def process_audio_file(audio_path, output_dir):
    try:
        # 生成唯一文件名（避免重复）
        filename = os.path.basename(audio_path)
        output_name = os.path.splitext(filename)[0] + '_mfcc.npy'
        output_path = os.path.join(output_dir, output_name)

        # 转换格式（如果是m4a则转wav）
        if audio_path.lower().endswith('.m4a'):
            audio = AudioSegment.from_file(audio_path)
            wav_path = os.path.join(output_dir, 'temp_converted.wav')
            audio.export(wav_path, format='wav')
            sampling_rate, signal = wavfile.read(wav_path)
            os.remove(wav_path)  # 删除临时文件
        else:
            sampling_rate, signal = wavfile.read(audio_path)

        # 确保单声道
        if len(signal.shape) > 1:
            signal = signal[:, 0]

        # 预加重滤波器
        pre_emphasis = 0.97
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        # 分帧
        frame_size = 0.025  # 25ms
        frame_stride = 0.01  # 10ms
        NFFT = 512
        num_mfcc = 13

        frame_length = int(round(frame_size * sampling_rate))
        frame_step = int(round(frame_stride * sampling_rate))
        signal_length = len(signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
        ).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # 加窗和FFT
        frames *= np.hamming(frame_length)
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = (1.0 / NFFT) * (mag_frames**2)

        # Mel滤波器组
        nfilt = 26
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sampling_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))
        bin = np.floor((NFFT + 1) * hz_points / sampling_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # 计算MFCC
        mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_mfcc]

        # 保存MFCC值
        np.save(output_path, mfcc)
        print(f"Successfully saved: {output_path}")

    except Exception as e:
        print(f"处理文件 {os.path.basename(audio_path)} 失败: {str(e)}")

if __name__ == "__main__":
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入文件夹
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in supported_formats):
            audio_path = os.path.join(input_dir, filename)
            process_audio_file(audio_path, output_dir)
