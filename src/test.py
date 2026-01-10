"""
Test script for Shazam implementation.

This script:
1. Builds a database from the MTG-Jamendo dataset
2. Creates synthetic degraded queries (noise, compression, etc.)
3. Tests the matching performance
"""

import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path
from shazam import ShazamSystem


def add_noise(audio, noise_level_db=-10):
    """
    Add white noise to audio signal.
    
    From paper: Tests with SNR from -15dB to +15dB
    
    Args:
        audio: Audio signal
        noise_level_db: Noise level in dB (negative = quieter than signal)
                       -10 dB means noise is 10dB quieter than signal
    
    Returns:
        Noisy audio signal
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power for desired SNR
    # SNR_dB = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / (10 ^ (SNR_dB / 10))
    noise_power = signal_power / (10 ** (noise_level_db / 10))
    
    # Generate white noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    # Add noise to signal
    noisy_audio = audio + noise
    
    return noisy_audio


def extract_segment(audio, Fs, duration_sec=10, start_sec=None):
    """
    Extract a segment from audio.
    
    From paper: Tests with 5, 10, and 15 second samples
    
    Args:
        audio: Full audio signal
        Fs: Sample rate
        duration_sec: Length of segment in seconds
        start_sec: Start time in seconds (None = random)
    
    Returns:
        Audio segment
    """
    total_duration = len(audio) / Fs
    
    # Random start if not specified
    if start_sec is None:
        max_start = total_duration - duration_sec
        if max_start > 0:
            start_sec = np.random.uniform(0, max_start)
        else:
            start_sec = 0
    
    # Extract segment
    start_sample = int(start_sec * Fs)
    end_sample = start_sample + int(duration_sec * Fs)
    
    segment = audio[start_sample:end_sample]
    
    return segment


def create_degraded_query(audio_path, output_path, 
                         duration_sec=10, 
                         noise_level_db=-6,
                         Fs=44100):
    """
    Create a degraded query sample from an audio file.
    
    Simulates the conditions described in the paper:
    - Extract a short segment (5-15 seconds)
    - Add noise (simulating environment noise)
    
    Args:
        audio_path: Path to source audio file
        output_path: Path to save degraded query
        duration_sec: Length of query in seconds
        noise_level_db: SNR in dB (negative = signal stronger than noise)
        Fs: Sample rate
    """
    # Load audio (fixed: use sr= as keyword argument)
    audio, sr = librosa.load(audio_path, sr=Fs)
    
    # Extract segment from middle or random position
    segment = extract_segment(audio, Fs, duration_sec)
    
    # Add noise
    noisy_segment = add_noise(segment, noise_level_db)
    
    # Normalize to prevent clipping
    noisy_segment = noisy_segment / np.max(np.abs(noisy_segment)) * 0.9
    
    # Save
    sf.write(output_path, noisy_segment, Fs)
    
    print(f"Created degraded query: {output_path}")
    print(f"  Duration: {duration_sec}s, SNR: {noise_level_db}dB")
    
    return output_path


def run_experiment(dataset_path, 
                   folders=['00', '01'],
                   num_test_queries=5,
                   duration_sec=10,
                   noise_level_db=-6):
    """
    Full experiment: build database and test queries.
    
    Args:
        dataset_path: Path to MTG-Jamendo dataset
        folders: Which folders to include in database
        num_test_queries: Number of test queries to create
        duration_sec: Duration of test queries
        noise_level_db: SNR for test queries
    """
    print("="*60)
    print("SHAZAM IMPLEMENTATION TEST")
    print("="*60)
    
    # Initialize system
    system = ShazamSystem()
    
    # Step 1: Build database
    print("\n[STEP 1] Building database...")
    system.build_database(dataset_path, folders=folders)
    
    # Save database
    db_path = 'shazam_database.pkl'
    system.save_database(db_path)
    
    # Step 2: Create test queries from random tracks in database
    print(f"\n[STEP 2] Creating {num_test_queries} test queries...")
    
    # Get random tracks from database
    track_ids = list(system.track_metadata.keys())
    test_track_ids = np.random.choice(track_ids, 
                                      size=min(num_test_queries, len(track_ids)), 
                                      replace=False)
    
    # Create output directory for queries
    query_dir = Path('test_queries')
    query_dir.mkdir(exist_ok=True)
    
    test_queries = []
    for i, track_id in enumerate(test_track_ids):
        source_path = system.track_metadata[track_id]
        query_path = query_dir / f"query_{i:02d}_track_{track_id}.wav"
        
        create_degraded_query(source_path, query_path, 
                            duration_sec=duration_sec,
                            noise_level_db=noise_level_db)
        
        test_queries.append((query_path, track_id))
    
    # Step 3: Test queries
    print(f"\n[STEP 3] Testing {len(test_queries)} queries...")
    print("="*60)
    
    correct = 0
    for query_path, true_track_id in test_queries:
        print(f"\nQuery: {query_path.name}")
        print(f"True track ID: {true_track_id}")
        
        # Query the database
        results = system.query(str(query_path), top_k=3)
        
        # Display results
        print("\nTop 3 matches:")
        for result in results:
            print(f"  Rank {result['rank']}: "
                  f"Track {result['track_id']} "
                  f"(score: {result['score']}, "
                  f"offset: {result['time_offset']} frames)")
        
        # Check if correct
        if results and results[0]['track_id'] == true_track_id:
            print("✓ CORRECT!")
            correct += 1
        else:
            print("✗ INCORRECT")
        
        print("-"*60)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total queries: {len(test_queries)}")
    print(f"Correct matches: {correct}")
    print(f"Accuracy: {correct/len(test_queries)*100:.1f}%")
    print(f"Database size: {len(system.track_metadata)} tracks")
    print(f"Total hashes: {len(system.database)}")


if __name__ == '__main__':
    # Configuration
    # UPDATED: Use your corrected path
    DATASET_PATH = './mtg-jamendo-dataset'
    
    # You can easily change which folders to include
    FOLDERS_TO_PROCESS = ['00'] 
    # To process all folders, use:
    # FOLDERS_TO_PROCESS = [f'{i:02d}' for i in range(100)]
    
    # Test parameters (from paper: 5, 10, 15 second samples at various SNRs)
    NUM_QUERIES = 10
    QUERY_DURATION = 10  # seconds
    NOISE_LEVEL = -6     # dB (paper tested -15 to +15 dB)
    
    # Run the experiment
    run_experiment(
        dataset_path=DATASET_PATH,
        folders=FOLDERS_TO_PROCESS,
        num_test_queries=NUM_QUERIES,
        duration_sec=QUERY_DURATION,
        noise_level_db=NOISE_LEVEL
    )
