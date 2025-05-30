import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio in dB"""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def spectral_subtraction(audio, sample_rate, noise_duration=0.1):
    """Noise removal using spectral subtraction"""
    # Extract noise
    noise = audio[:int(noise_duration * sample_rate)]
    noise_spectrum = np.abs(np.fft.fft(noise))
    noise_spectrum = gaussian_filter(noise_spectrum, sigma=1)
    
    # Process in frames
    frame_size = int(0.02 * sample_rate)
    overlap = int(0.7 * frame_size)
    frames = [audio[i:i+frame_size] for i in range(0, len(audio) - frame_size, frame_size - overlap)]
    spectrums = [np.fft.fft(frame) for frame in frames]
    
    # Apply spectral subtraction
    clean_spectrums = []
    for spectrum in spectrums:
        magnitude = np.abs(spectrum) - noise_spectrum[:len(spectrum)]
        magnitude = np.maximum(magnitude, 1e-6)  # Avoid complete suppression
        clean_spectrum = magnitude * np.exp(1j * np.angle(spectrum))
        clean_spectrums.append(clean_spectrum)
    
    # Overlap-add synthesis
    clean_frames = [np.fft.ifft(spectrum).real for spectrum in clean_spectrums]
    clean_audio = np.zeros(len(audio))
    for i, frame in enumerate(clean_frames):
        start = i * (frame_size - overlap)
        clean_audio[start:start+frame_size] += frame
    
    return clean_audio

def wiener_filter(audio, sample_rate, noise_duration=0.1, alpha=1.0):
    """Noise removal using Wiener filtering"""
    # Extract noise
    noise = audio[:int(noise_duration * sample_rate)]
    noise_spectrum = np.abs(np.fft.fft(noise))**2
    
    # Process in frames
    frame_size = int(0.02 * sample_rate)
    overlap = int(0.7 * frame_size)
    frames = [audio[i:i+frame_size] for i in range(0, len(audio) - frame_size, frame_size - overlap)]
    spectrums = [np.fft.fft(frame) for frame in frames]
    
    # Apply Wiener filter
    clean_spectrums = []
    window = np.hanning(frame_size)
    for spectrum in spectrums:
        signal_power = np.abs(spectrum)**2
        wiener = signal_power / (signal_power + alpha * noise_spectrum[:len(spectrum)])
        clean_spectrum = spectrum * wiener
        clean_spectrums.append(clean_spectrum)
    
    # Overlap-add synthesis with window
    clean_audio = np.zeros(len(audio))
    for i, frame in enumerate(clean_spectrums):
        start = i * (frame_size - overlap)
        clean_audio[start:start+frame_size] += np.fft.ifft(frame).real * window
    
    return clean_audio

def hybrid_filter(audio, sample_rate, noise_duration=0.1, blend_factor=0.5, alpha=1.0):
    """
    Combine spectral subtraction and Wiener filtering
    blend_factor: 0 = pure spectral subtraction, 1 = pure Wiener filter
    """
    # Get the output from both methods
    ss_output = spectral_subtraction(audio, sample_rate, noise_duration)
    wf_output = wiener_filter(audio, sample_rate, noise_duration, alpha)
    
    # Blend the results
    return ss_output * (1 - blend_factor) + wf_output * blend_factor

def adaptive_filter(audio, sample_rate):
    """
    Adaptively select the best method based on initial SNR analysis
    """
    noise = audio[:int(0.1 * sample_rate)]
    signal = audio[int(0.1 * sample_rate):]
    initial_snr = calculate_snr(signal, noise)
    
    # Adjust parameters based on SNR
    if initial_snr < 0:  # Very noisy
        blend_factor = 0.7  # Favor Wiener filter for very noisy signals
        alpha = 2.0  # More aggressive noise removal
    elif initial_snr < 10:  # Moderately noisy
        blend_factor = 0.5  # Equal blend
        alpha = 1.0  # Standard noise removal
    else:  # Less noisy
        blend_factor = 0.3  # Favor spectral subtraction for cleaner signals
        alpha = 0.5  # Less aggressive noise removal
    
    print(f"Initial SNR: {initial_snr:.2f} dB")
    print(f"Selected blend factor: {blend_factor:.2f}, alpha: {alpha:.2f}")
    
    return hybrid_filter(audio, sample_rate, 0.1, blend_factor, alpha), blend_factor, alpha

def main():
    # Read audio file
    sample_rate, audio = wavfile.read(r"D:\ISDN2002\testing\input\Audio.wav")
    
    # Convert to mono if stereo
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
    
    # Extract noise segment for SNR calculation
    noise = audio[:int(0.1 * sample_rate)]
    signal = audio[int(0.1 * sample_rate):]
    initial_snr = calculate_snr(signal, noise)
    
    print("Processing with different methods...")
    
    # Process with different methods
    ss_output = spectral_subtraction(audio, sample_rate)
    wf_output = wiener_filter(audio, sample_rate)
    
    # Try different blend factors
    blend_outputs = {}
    for blend in [0.0, 0.3, 0.5, 0.7, 1.0]:
        blend_outputs[blend] = hybrid_filter(audio, sample_rate, 0.1, blend)
    
    # Adaptive method
    adaptive_output, blend_used, alpha_used = adaptive_filter(audio, sample_rate)
    
    # Calculate SNRs
    ss_snr = calculate_snr(ss_output[int(0.1 * sample_rate):], noise)
    wf_snr = calculate_snr(wf_output[int(0.1 * sample_rate):], noise)
    adaptive_snr = calculate_snr(adaptive_output[int(0.1 * sample_rate):], noise)
    blend_snrs = {blend: calculate_snr(output[int(0.1 * sample_rate):], noise) 
                 for blend, output in blend_outputs.items()}
    
    # Print results
    print(f"Original SNR: {initial_snr:.2f} dB")
    print(f"Spectral Subtraction SNR: {ss_snr:.2f} dB (improvement: {ss_snr - initial_snr:.2f} dB)")
    print(f"Wiener Filter SNR: {wf_snr:.2f} dB (improvement: {wf_snr - initial_snr:.2f} dB)")
    print(f"Adaptive method SNR: {adaptive_snr:.2f} dB (improvement: {adaptive_snr - initial_snr:.2f} dB)")
    
    print("\nBlend factor results:")
    for blend, snr in blend_snrs.items():
        print(f"  Blend {blend:.1f}: {snr:.2f} dB (improvement: {snr - initial_snr:.2f} dB)")
    
    # Find best method
    all_snrs = [ss_snr, wf_snr, adaptive_snr] + list(blend_snrs.values())
    best_snr = max(all_snrs)
    methods = ["Spectral Subtraction", "Wiener Filter", "Adaptive"] + [f"Blend {b:.1f}" for b in blend_outputs.keys()]
    best_method = methods[all_snrs.index(best_snr)]
    
    print(f"\nBest method: {best_method} with SNR: {best_snr:.2f} dB")
    
    # Save best result
    if best_method == "Spectral Subtraction":
        best_output = ss_output
    elif best_method == "Wiener Filter":
        best_output = wf_output
    elif best_method == "Adaptive":
        best_output = adaptive_output
    else:
        blend = float(best_method.split()[1])
        best_output = blend_outputs[blend]
    
    # Normalize and save
    best_output = best_output / np.max(np.abs(best_output))
    best_output = (best_output * 32767).astype(np.int16)
    wavfile.write("enhanced_audio.wav", sample_rate, best_output)
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Time domain comparison
    plt.subplot(3, 1, 1)
    plt.plot(audio)
    plt.title(f"Original Audio (SNR: {initial_snr:.2f} dB)")
    
    plt.subplot(3, 1, 2)
    plt.plot(ss_output, label=f"Spectral Subtraction (SNR: {ss_snr:.2f} dB)")
    plt.plot(wf_output, alpha=0.6, label=f"Wiener Filter (SNR: {wf_snr:.2f} dB)")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(best_output)
    plt.title(f"Best Method: {best_method} (SNR: {best_snr:.2f} dB)")
    
    plt.tight_layout()
    plt.savefig("noise_removal_comparison.png")
    plt.show()
    
    # Frequency domain visualization
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.magnitude_spectrum(audio, Fs=sample_rate, scale='dB')
    plt.title("Original Audio Spectrum")
    
    plt.subplot(2, 1, 2)
    plt.magnitude_spectrum(best_output, Fs=sample_rate, scale='dB')
    plt.title(f"Enhanced Audio Spectrum ({best_method})")
    
    plt.tight_layout()
    plt.savefig("spectrum_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()