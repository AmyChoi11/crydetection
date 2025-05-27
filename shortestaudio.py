import os
import time
import soundfile as sf
from tqdm import tqdm
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def get_audio_duration(file_path):
    """Get audio duration using soundfile (faster than librosa)"""
    try:
        info = sf.info(file_path)
        return info.duration, file_path
    except Exception as e:
        return None, None

def find_audio_durations(root_dir, max_workers=4):
    """Find durations of all audio files in a directory and its subdirectories"""
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    shortest_file = None
    shortest_duration = float('inf')
    all_durations = []  # Store all durations for histogram
    
    # Find all audio files first
    print("Finding all audio files...")
    audio_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                # Only add files that exist and are not too large (100MB limit)
                file_path = os.path.join(root, file)
                try:
                    if os.path.getsize(file_path) < 100_000_000:  # 100MB limit
                        audio_files.append(file_path)
                except:
                    pass
    
    print(f"Found {len(audio_files)} audio files. Analyzing durations...")
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_audio_duration, file_path): file_path for file_path in audio_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(audio_files)):
            file_path = futures[future]
            try:
                duration, path = future.result()
                if duration is not None:
                    all_durations.append(duration)
                    if duration < shortest_duration:
                        shortest_duration = duration
                        shortest_file = path
                        print(f"\nNew shortest found: {os.path.basename(path)} ({duration:.2f}s)")
            except Exception as e:
                print(f"\nError processing {file_path}: {e}")
    
    # Calculate frames based on your model parameters
    frames = int(shortest_duration * 22050 / 256) if shortest_duration != float('inf') else 0
    
    return shortest_file, shortest_duration, frames, all_durations

def plot_duration_histogram(durations):
    """Plot histogram of audio durations"""
    if not durations:
        return
    
    # Convert to NumPy array for statistics
    durations = np.array(durations)
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    # Main histogram
    n, bins, patches = plt.hist(durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical line for mean
    plt.axvline(x=np.mean(durations), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(durations):.2f}s')
    
    # Add vertical line for median
    plt.axvline(x=np.median(durations), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(durations):.2f}s')
    
    # Format x-axis in seconds
    def seconds_formatter(x, pos):
        return f'{x:.1f}s'
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(seconds_formatter))
    
    # Add labels and title
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Audio Files')
    plt.title('Distribution of Audio File Durations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text with statistics
    stats_text = (
        f"Total files: {len(durations)}\n"
        f"Min duration: {np.min(durations):.2f}s\n"
        f"Max duration: {np.max(durations):.2f}s\n"
        f"Mean duration: {np.mean(durations):.2f}s\n"
        f"Median duration: {np.median(durations):.2f}s\n"
        f"Std deviation: {np.std(durations):.2f}s"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.savefig('audio_duration_histogram.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    shortest_file, duration, frames, all_durations = find_audio_durations(r"D:\ISDN2002\archive")
    
    print("\n" + "="*50)
    if shortest_file:
        print(f"Shortest audio file: {shortest_file}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"This would be approximately {frames} frames")
        print(f"(using SR=22050, hop_length=256)")
        
        # Calculate stats
        if all_durations:
            durations = np.array(all_durations)
            print("\nAudio Duration Statistics:")
            print(f"Total files: {len(durations)}")
            print(f"Mean duration: {np.mean(durations):.2f}s")
            print(f"Median duration: {np.median(durations):.2f}s")
            print(f"Min duration: {np.min(durations):.2f}s")
            print(f"Max duration: {np.max(durations):.2f}s")
            print(f"Standard deviation: {np.std(durations):.2f}s")
            
            # Plot histogram
            print("\nGenerating histogram...")
            plot_duration_histogram(all_durations)
    else:
        print("No audio files found.")
        
    print(f"\nSearch completed in {time.time() - start_time:.2f} seconds")
    print("="*50)