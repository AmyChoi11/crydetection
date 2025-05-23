class Config:
    CLASS_LABELS = [
        'unwelltired', 'hungry', 'others', 'laugh',
        'quiet'
    ]
    NUM_CLASSES = len(CLASS_LABELS)
    BATCH_SIZE = 32  # Reduced to handle larger sequences
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MIN_SAMPLES = 120
    MAX_FRAMES = 431  # = 5 seconds * 22050 Hz / 256 hop_length
    N_MFCC = 24
    SAMPLE_RATE = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    # Update Config class to process 5 seconds of audio
    class Config:
        CLASS_LABELS = [
            'unwelltired', 'hungry', 'others', 'laugh',
            'quiet'
        ]
        NUM_CLASSES = len(CLASS_LABELS)
        BATCH_SIZE = 32  # Reduced batch size to handle larger sequences
        EPOCHS = 50
        LEARNING_RATE = 0.001
        MIN_SAMPLES = 120
        # Calculate frames for 5 seconds: 5 sec * 22050 Hz / 256 hop_length ≈ 431
        MAX_FRAMES = 431  # Updated from 48 to ~431 frames (5 seconds)
        N_MFCC = 24
        SAMPLE_RATE = 22050
        N_FFT = 1024
        HOP_LENGTH = 256