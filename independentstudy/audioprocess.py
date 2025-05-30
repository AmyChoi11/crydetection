import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import warnings
import argparse
from datetime import datetime
import shutil

# Import PESQ
try:
    from pesq import pesq
except ImportError:
    print("PESQ library not found. Installing...")
    import pip
    pip.main(['install', 'pesq'])
    from pesq import pesq

# Custom modules 
from combinemethod import spectral_subtraction, wiener_filter, hybrid_filter, adaptive_filter, calculate_snr

# Import the awgn function if it's in a separate module
try:
    from awgn import awgn
except ImportError:
    # Define a simple version if import fails
    def awgn(signal, snr, out='signal', method='vectorized', axis=0):
        """Add white Gaussian noise to the signal."""
        # Calculate signal power
        sig_power = np.mean(signal**2)
        # Calculate noise power based on SNR
        noise_power = sig_power / (10**(snr/10))
        # Generate white noise
        noise = np.sqrt(noise_power) * np.random.normal(0, 1, signal.shape)
        # Return noisy signal
        return signal + noise

# Suppress warnings
warnings.filterwarnings('ignore')

# Define paths
BASE_AUDIO_PATH = r"D:\ISDN2002\archive\Cry\1-60997-A.wav"
NOISE_DIR = r"D:\ISDN2002\noise"
OUTPUT_DIR = r"D:\ISDN2002\independentstudy\combinedaudios"
IMAGES_DIR = r"D:\ISDN2002\independentstudy\images"
RESULTS_FILE = r"D:\ISDN2002\independentstudy\noise_removal_results.txt"
PESQ_RESULTS_FILE = r"D:\ISDN2002\independentstudy\pesq_results.txt"

def calculate_pesq(clean_audio, processed_audio, sample_rate):
    """
    Calculate PESQ score between clean and processed audio
    
    Args:
        clean_audio: Original clean audio signal
        processed_audio: Processed/enhanced audio signal
        sample_rate: Sample rate of the audio signals
        
    Returns:
        PESQ score (higher is better)
    """
    # PESQ requires a specific sample rate (8000 or 16000 Hz)
    target_sr = 16000
    
    # Resample if needed
    if sample_rate != target_sr:
        clean_audio = signal.resample(clean_audio, int(len(clean_audio) * target_sr / sample_rate))
        processed_audio = signal.resample(processed_audio, int(len(processed_audio) * target_sr / sample_rate))
    
    # Ensure amplitude range is [-1, 1]
    clean_audio = clean_audio / np.max(np.abs(clean_audio))
    processed_audio = processed_audio / np.max(np.abs(processed_audio))
    
    # Calculate PESQ score (wb = wideband)
    try:
        score = pesq(target_sr, clean_audio, processed_audio, 'wb')
        return score
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        return -1

def clear_output_directories(images_dir, output_dir, results_file=None, confirm=True):
    """
    Clear previously generated files from output directories.
    
    Args:
        images_dir: Directory containing generated images
        output_dir: Directory containing processed audio files
        results_file: Results text file to clear (optional)
        confirm: Whether to ask for confirmation before deleting (default: True)
    """
    if confirm:
        print("\nWARNING: This will delete all previously generated files!")
        response = input("Do you want to continue? (y/n): ").strip().lower()
        if response != 'y':
            print("Cleanup cancelled. Continuing with processing...")
            return
    
    # Clear images directory
    print(f"Clearing images directory: {images_dir}")
    image_count = 0
    for file in glob.glob(os.path.join(images_dir, "*_comparison.png")):
        os.remove(file)
        image_count += 1
    for file in glob.glob(os.path.join(images_dir, "*_spectrum.png")):
        os.remove(file)
        image_count += 1
    print(f"Removed {image_count} image files")
    
    # Clear audio output directory
    print(f"Clearing output audio directory: {output_dir}")
    audio_count = 0
    for file in glob.glob(os.path.join(output_dir, "*.wav")):
        os.remove(file)
        audio_count += 1
    print(f"Removed {audio_count} audio files")
    
    # Clear results files if specified
    if results_file and os.path.exists(results_file):
        print(f"Clearing results file: {results_file}")
        open(results_file, 'w', encoding='utf-8').close()  # Just clear the file content
        
    # Clear PESQ results file
    pesq_file = r"D:\ISDN2002\independentstudy\pesq_results.txt"
    if os.path.exists(pesq_file):
        print(f"Clearing PESQ results file: {pesq_file}")
        open(pesq_file, 'w', encoding='utf-8').close()
    
    print("Cleanup completed.")

def combine_audio(clean_path, noise_path, output_path, snr_target=5):
    """Mix clean audio with noise at specified SNR"""
    # Read files
    sr_clean, clean = wavfile.read(clean_path)
    sr_noise, noise = wavfile.read(noise_path)
    
    # Convert to mono if needed
    if len(clean.shape) > 1:
        clean = np.mean(clean, axis=1).astype(np.float32)
    else:
        clean = clean.astype(np.float32)
        
    if len(noise.shape) > 1:
        noise = np.mean(noise, axis=1).astype(np.float32)
    else:
        noise = noise.astype(np.float32)
    
    # Normalize to [-1, 1]
    clean = clean / np.max(np.abs(clean))
    noise = noise / np.max(np.abs(noise))
    
    # If noise is shorter, repeat it
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean)]
    else:
        # If noise is longer, truncate it
        noise = noise[:len(clean)]
    
    # Calculate scaling factor for desired SNR
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    scaling_factor = np.sqrt(clean_power / (10**(snr_target/10) * noise_power))
    
    # Scale the noise and add to clean signal
    scaled_noise = noise * scaling_factor
    mixed = clean + scaled_noise
    
    # Normalize to prevent clipping
    mixed = mixed / np.max(np.abs(mixed))
    
    # Convert to int16
    mixed_int16 = (mixed * 32767).astype(np.int16)
    
    # Save mixed audio
    wavfile.write(output_path, sr_clean, mixed_int16)
    
    # Calculate actual SNR
    actual_snr = calculate_snr(clean, scaled_noise)
    
    return sr_clean, mixed, actual_snr

def add_white_noise(clean_path, output_path, snr_target=5):
    """Add synthetic white noise to audio at specified SNR"""
    # Read clean audio
    sr_clean, clean = wavfile.read(clean_path)
    
    # Convert to mono if needed
    if len(clean.shape) > 1:
        clean = np.mean(clean, axis=1).astype(np.float32)
    else:
        clean = clean.astype(np.float32)
    
    # Normalize to [-1, 1]
    clean = clean / np.max(np.abs(clean))
    
    # Add white Gaussian noise using awgn function
    noisy = awgn(clean, snr_target, out='signal', method='vectorized', axis=0)
    
    # Generate the actual noise component
    noise = noisy - clean
    
    # Normalize to prevent clipping
    noisy = noisy / np.max(np.abs(noisy))
    
    # Convert to int16
    noisy_int16 = (noisy * 32767).astype(np.int16)
    
    # Save noisy audio
    wavfile.write(output_path, sr_clean, noisy_int16)
    
    # Calculate actual SNR
    actual_snr = calculate_snr(clean, noise)
    
    return sr_clean, noisy, actual_snr

def process_audio(noisy_path, noise_type, original_snr):
    """Process noisy audio using combinemethod and save results"""
    # Load the noisy audio
    sample_rate, audio = wavfile.read(noisy_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Normalize
    audio = audio.astype(np.float32) / 32767
    
    # Extract filename for output naming
    base_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{base_filename}_{timestamp}"
    
    # Extract noise segment for SNR calculation
    noise = audio[:int(0.1 * sample_rate)]
    signal = audio[int(0.1 * sample_rate):]
    initial_snr = calculate_snr(signal, noise)
    
    print(f"Processing {base_filename}...")
    print(f"Original SNR: {initial_snr:.2f} dB")
    
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
    
    # Find best method based on SNR
    all_snrs = [ss_snr, wf_snr, adaptive_snr] + list(blend_snrs.values())
    best_snr = max(all_snrs)
    methods = ["Spectral Subtraction", "Wiener Filter", "Adaptive"] + [f"Blend {b:.1f}" for b in blend_outputs.keys()]
    best_method_snr = methods[all_snrs.index(best_snr)]
    
    # Read the original clean audio for PESQ comparison
    _, clean_audio = wavfile.read(BASE_AUDIO_PATH)
    if len(clean_audio.shape) > 1:
        clean_audio = np.mean(clean_audio, axis=1)
    clean_audio = clean_audio.astype(np.float32) / 32767
    
    # Make sure the lengths match (use the shorter length)
    min_length = min(len(clean_audio), len(audio))
    clean_audio = clean_audio[:min_length]
    audio_trimmed = audio[:min_length]
    
    # Calculate PESQ scores for each method
    print("Calculating PESQ scores...")
    pesq_ss = calculate_pesq(clean_audio, ss_output[:min_length], sample_rate)
    pesq_wf = calculate_pesq(clean_audio, wf_output[:min_length], sample_rate)
    pesq_adaptive = calculate_pesq(clean_audio, adaptive_output[:min_length], sample_rate)
    
    # Calculate PESQ for each blend factor
    pesq_blend = {}
    for blend, output in blend_outputs.items():
        pesq_blend[blend] = calculate_pesq(clean_audio, output[:min_length], sample_rate)
    
    # Find best method by PESQ
    all_pesq = [pesq_ss, pesq_wf, pesq_adaptive] + list(pesq_blend.values())
    best_pesq = max(all_pesq)
    methods_pesq = ["Spectral Subtraction", "Wiener Filter", "Adaptive"] + [f"Blend {b:.1f}" for b in blend_outputs.keys()]
    best_method_pesq = methods_pesq[all_pesq.index(best_pesq)]
    
    # CHANGE: Use PESQ-optimized method as the "best method" for output
    best_method = best_method_pesq
    
    # Select best output based on PESQ instead of SNR
    if best_method == "Spectral Subtraction":
        best_output = ss_output
    elif best_method == "Wiener Filter":
        best_output = wf_output
    elif best_method == "Adaptive":
        best_output = adaptive_output
    else:
        blend = float(best_method.split()[1])
        best_output = blend_outputs[blend]
    
    # Normalize and save best result
    best_output = best_output / np.max(np.abs(best_output))
    best_output = (best_output * 32767).astype(np.int16)
    enhanced_file = os.path.join(OUTPUT_DIR, f"{output_prefix}_enhanced_PESQ.wav")
    wavfile.write(enhanced_file, sample_rate, best_output)
    
    # Plot time domain comparison
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(audio)
    plt.title(f"Original Audio (SNR: {initial_snr:.2f} dB)")
    
    plt.subplot(3, 1, 2)
    plt.plot(ss_output, label=f"Spectral Subtraction (SNR: {ss_snr:.2f} dB, PESQ: {pesq_ss:.2f})")
    plt.plot(wf_output, alpha=0.6, label=f"Wiener Filter (SNR: {wf_snr:.2f} dB, PESQ: {pesq_wf:.2f})")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(best_output)
    plt.title(f"Best Method (PESQ): {best_method} (PESQ: {best_pesq:.2f}, SNR: {all_snrs[all_pesq.index(best_pesq)]:.2f} dB)")
    
    plt.tight_layout()
    comparison_img = os.path.join(IMAGES_DIR, f"{output_prefix}_comparison.png")
    plt.savefig(comparison_img)
    plt.close()
    
    # Plot frequency domain comparison
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.magnitude_spectrum(audio, Fs=sample_rate, scale='dB')
    plt.title("Original Audio Spectrum")
    
    plt.subplot(2, 1, 2)
    plt.magnitude_spectrum(best_output, Fs=sample_rate, scale='dB')
    plt.title(f"Enhanced Audio Spectrum ({best_method}, optimized for PESQ)")
    
    plt.tight_layout()
    spectrum_img = os.path.join(IMAGES_DIR, f"{output_prefix}_spectrum.png")
    plt.savefig(spectrum_img)
    plt.close()
    
    # Collect results
    results = {
        "file": os.path.basename(noisy_path),
        "noise_type": noise_type,
        "target_snr": original_snr,
        "initial_snr": initial_snr,
        "ss_snr": ss_snr,
        "wf_snr": wf_snr,
        "adaptive_snr": adaptive_snr,
        "blend_snrs": blend_snrs,
        "best_method_snr": best_method_snr,
        "best_snr": best_snr,
        "improvement_snr": best_snr - initial_snr,
        # PESQ results
        "pesq_ss": pesq_ss,
        "pesq_wf": pesq_wf,
        "pesq_adaptive": pesq_adaptive,
        "pesq_blend": pesq_blend,
        "best_method_pesq": best_method_pesq,
        "best_pesq": best_pesq,
        "best_method": best_method  # This is now the PESQ-based best method
    }
    
    return results

def write_results_to_file(results_list, filename):
    """Write processing results to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Noise Removal Processing Results\n")
        f.write("===============================\n\n")
        
        for result in results_list:
            f.write(f"File: {result['file']}\n")
            f.write(f"Noise Type: {result['noise_type']}\n")
            f.write(f"Target SNR: {result['target_snr']:.2f} dB\n")
            f.write(f"Measured Initial SNR: {result['initial_snr']:.2f} dB\n")
            
            # Technical measurements (SNR)
            f.write("\nSNR Measurements:\n")
            f.write(f"Spectral Subtraction: {result['ss_snr']:.2f} dB (improvement: {result['ss_snr'] - result['initial_snr']:.2f} dB)\n")
            f.write(f"Wiener Filter: {result['wf_snr']:.2f} dB (improvement: {result['wf_snr'] - result['initial_snr']:.2f} dB)\n")
            f.write(f"Adaptive Method: {result['adaptive_snr']:.2f} dB (improvement: {result['adaptive_snr'] - result['initial_snr']:.2f} dB)\n")
            
            f.write("\nBlend Factor SNR Results:\n")
            for blend, snr in result['blend_snrs'].items():
                f.write(f"  Blend {blend:.1f}: {snr:.2f} dB (improvement: {snr - result['initial_snr']:.2f} dB)\n")
            
            # Perceptual quality measurements (PESQ)
            f.write("\nPESQ Measurements:\n")
            f.write(f"Spectral Subtraction: {result['pesq_ss']:.2f}\n")
            f.write(f"Wiener Filter: {result['pesq_wf']:.2f}\n")
            f.write(f"Adaptive Method: {result['pesq_adaptive']:.2f}\n")
            
            f.write("\nBlend Factor PESQ Results:\n")
            for blend, pesq_score in result['pesq_blend'].items():
                f.write(f"  Blend {blend:.1f}: {pesq_score:.2f}\n")
            
            # Combined results
            f.write(f"\nBest Method by SNR: {result['best_method_snr']} with SNR: {result['best_snr']:.2f} dB\n")
            f.write(f"Best Method by PESQ: {result['best_method_pesq']} with PESQ: {result['best_pesq']:.2f}\n")
            f.write(f"\n*** Selected Method: {result['best_method']} (optimized for perceptual quality) ***\n\n")
            f.write("------------------------------------------------\n\n")

def write_pesq_results_to_file(results_list, filename):
    """Write PESQ results to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("PESQ Evaluation Results\n")
        f.write("=====================\n\n")
        
        f.write("Legend: PESQ scores range from -0.5 to 4.5, where higher is better.\n")
        f.write("  - 4.5: Excellent quality, imperceptible differences\n")
        f.write("  - 4.0: Very good quality, perceptible but not annoying\n")
        f.write("  - 3.0: Good quality, slightly annoying\n")
        f.write("  - 2.0: Fair quality, annoying\n")
        f.write("  - 1.0: Poor quality, very annoying\n\n")
        
        # Group by noise type and SNR level
        noise_types = set(result['noise_type'] for result in results_list)
        snr_levels = sorted(set(result['target_snr'] for result in results_list))
        
        # Summary table header
        f.write("Summary Table\n")
        f.write("------------\n\n")
        f.write("| Noise Type | SNR | Spectral Sub. | Wiener Filter | Adaptive | Best Method | Best PESQ |\n")
        f.write("|------------|-----|--------------|---------------|----------|------------|----------|\n")
        
        for noise in sorted(noise_types):
            for snr in snr_levels:
                matching_results = [r for r in results_list if r['noise_type'] == noise and r['target_snr'] == snr]
                
                if matching_results:
                    result = matching_results[0]
                    f.write(f"| {noise} | {snr} dB | {result['pesq_ss']:.2f} | {result['pesq_wf']:.2f} | ")
                    f.write(f"{result['pesq_adaptive']:.2f} | {result['best_method_pesq']} | {result['best_pesq']:.2f} |\n")
        
        f.write("\n\nDetailed Results\n")
        f.write("---------------\n\n")
        
        for result in results_list:
            f.write(f"File: {result['file']}\n")
            f.write(f"Noise Type: {result['noise_type']}\n")
            f.write(f"Target SNR: {result['target_snr']:.2f} dB\n")
            f.write(f"Measured Initial SNR: {result['initial_snr']:.2f} dB\n\n")
            
            f.write(f"Spectral Subtraction PESQ: {result['pesq_ss']:.2f}\n")
            f.write(f"Wiener Filter PESQ: {result['pesq_wf']:.2f}\n")
            f.write(f"Adaptive Method PESQ: {result['pesq_adaptive']:.2f}\n")
            
            f.write("\nBlend Factor PESQ Results:\n")
            for blend, pesq_score in result['pesq_blend'].items():
                f.write(f"  Blend {blend:.1f}: {pesq_score:.2f}\n")
            
            f.write(f"\nBest Method (PESQ): {result['best_method_pesq']} with PESQ: {result['best_pesq']:.2f}\n")
            f.write(f"Best Method (SNR): {result['best_method_snr']} with SNR: {result['best_snr']:.2f} dB\n")
            
            if result['best_method_pesq'] == result['best_method_snr']:
                f.write("Note: SNR and PESQ metrics agree on the best method.\n")
            else:
                f.write("Note: SNR and PESQ metrics suggest different optimal methods.\n")
                f.write(f"      PESQ-optimized method ({result['best_method_pesq']}) was selected for output.\n")
                
            f.write("\n------------------------------------------------\n\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio noise processing pipeline")
    parser.add_argument("--clean", action="store_true", help="Clean output directories before processing")
    parser.add_argument("--force-clean", action="store_true", help="Clean output directories without confirmation")
    args = parser.parse_args()
    
    print("Starting audio noise processing pipeline...")
    
    # Create directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Clean if requested
    if args.clean or args.force_clean:
        clear_output_directories(IMAGES_DIR, OUTPUT_DIR, RESULTS_FILE, confirm=not args.force_clean)
    
    # Get list of noise files
    noise_files = glob.glob(os.path.join(NOISE_DIR, "*.wav"))
    
    if not noise_files:
        print(f"No noise files found in {NOISE_DIR}")
        return
    
    print(f"Found {len(noise_files)} noise files")
    
    # Check if base audio file exists
    if not os.path.exists(BASE_AUDIO_PATH):
        print(f"Base audio file not found: {BASE_AUDIO_PATH}")
        return
    
    # Process each noise file
    results_list = []
    
    # Define SNR levels to test
    target_snrs = [0, 5, 10]
    
    # First process with recorded noise files
    for noise_file in noise_files:
        noise_name = os.path.splitext(os.path.basename(noise_file))[0]
        print(f"\nProcessing noise: {noise_name}")
        
        for snr in target_snrs:
            # Combine audio with noise
            output_name = f"{noise_name}_SNR{snr}dB.wav"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            
            print(f"Mixing with target SNR: {snr} dB")
            _, _, actual_snr = combine_audio(BASE_AUDIO_PATH, noise_file, output_path, snr_target=snr)
            print(f"Actual SNR: {actual_snr:.2f} dB")
            
            # Process the noisy audio
            results = process_audio(output_path, noise_name, snr)
            results_list.append(results)
    
    # Then process with white noise
    print("\nProcessing with synthetic white noise")
    for snr in target_snrs:
        # Add white noise to audio
        output_name = f"white_noise_SNR{snr}dB.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        print(f"Adding white noise with target SNR: {snr} dB")
        _, _, actual_snr = add_white_noise(BASE_AUDIO_PATH, output_path, snr_target=snr)
        print(f"Actual SNR: {actual_snr:.2f} dB")
        
        # Process the noisy audio
        results = process_audio(output_path, "white_noise", snr)
        results_list.append(results)
    
    # Write results to text files
    write_results_to_file(results_list, RESULTS_FILE)
    write_pesq_results_to_file(results_list, PESQ_RESULTS_FILE)
    
    print(f"\nAll processing complete!")
    print(f"- Noisy audio files saved to: {OUTPUT_DIR}")
    print(f"- Enhanced audio files (optimized for PERCEPTUAL QUALITY) saved to: {OUTPUT_DIR}")
    print(f"- Analysis images saved to: {IMAGES_DIR}")
    print(f"- SNR results summary saved to: {RESULTS_FILE}")
    print(f"- PESQ results summary saved to: {PESQ_RESULTS_FILE}")

if __name__ == "__main__":
    main()