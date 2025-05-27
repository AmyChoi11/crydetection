import os
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
import scipy.fftpack
from tqdm import tqdm

# ================== 参数配置 ==================
base_input_dir = r"D:\ISDN2002\archive"  # 输入音频文件夹路径
output_dir = r"D:\ISDN2002\mfcc_value"   # 输出MFCC值保存路径
supported_formats = ['.wav', '.m4a', '.ogg']  # 支持的音频格式
# =============================================
def process_audio_file(audio_path, category_folder):
    try:
        # Generate unique filename
        filename = os.path.basename(audio_path)
        output_name = os.path.splitext(filename)[0] + '_mfcc.npy'
        output_path = os.path.join(category_folder, output_name)

        print(f"Processing: {audio_path}")
        print(f"Saving to: {output_path}")

        # Load audio file
        if audio_path.lower().endswith('.m4a'):
            audio = AudioSegment.from_file(audio_path)
            wav_path = os.path.join(os.path.dirname(output_path), 'temp_converted.wav')
            audio.export(wav_path, format='wav')
            sampling_rate, signal = wavfile.read(wav_path)
            os.remove(wav_path)  # Delete temp file
        else:
            sampling_rate, signal = wavfile.read(audio_path)

        # Convert to mono if stereo
        if len(signal.shape) > 1:
            signal = signal[:, 0]
            
        # Convert to float32 if needed
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32) / np.max(np.abs(signal))

        # Pre-emphasis filter
        pre_emphasis = 0.97
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        # Frame parameters
        frame_size = 0.025  # 25ms
        frame_stride = 0.01  # 10ms
        frame_length = int(round(frame_size * sampling_rate))
        frame_step = int(round(frame_stride * sampling_rate))
        signal_length = len(signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        # Pad signal
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)

        # Frame the signal
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # Apply window
        frames *= np.hamming(frame_length)

        # FFT
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

        # Filter banks
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (sampling_rate / 2) / 700)
        nfilt = 40
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
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

        # MFCCs
        num_mfcc = 13
        mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_mfcc + 1)]
        
        # Make the shape uniform for our model (transpose to time x coefficients)
        mfcc = mfcc.T
        
        # Save MFCC values to category folder
        np.save(output_path, mfcc)
        print(f"Successfully saved: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {str(e)}")
        return False
    
if __name__ == "__main__":
    # Define your categories
    categories = ["Unwell", "Sleeping", "Cry", "Laugh", "Tired", "Silence"]
    
    # Create all category folders
    for category in categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    # Process files from each category subfolder
    success_count = 0
    error_count = 0
    
    for category in categories:
        # Input subfolder for this category
        category_input = os.path.join(base_input_dir, category)
        # Output subfolder for this category
        category_output = os.path.join(output_dir, category)
        
        if os.path.exists(category_input):
            print(f"\nProcessing category: {category}")
            for filename in os.listdir(category_input):
                if any(filename.lower().endswith(ext) for ext in supported_formats):
                    audio_path = os.path.join(category_input, filename)
                    if process_audio_file(audio_path, category_output):
                        success_count += 1
                    else:
                        error_count += 1
        else:
            print(f"Warning: Category folder not found: {category_input}")
    
    print(f"\nProcessing complete! Successful: {success_count}, Errors: {error_count}")