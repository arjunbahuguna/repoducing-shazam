"""
Utility functions for analyzing and visualizing Shazam results.

This recreates the analysis figures from the Wang03 paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from shazam import ShazamSystem
from collections import defaultdict


def visualize_scatterplot(system, query_path, track_id, save_path=None):
    """
    Recreate Figure 2A and 3A from the paper: scatterplot of time pairs.
    
    This shows the relationship between query time and database time for 
    matching hashes. A correct match shows a diagonal line.
    
    Args:
        system: ShazamSystem instance with loaded database
        query_path: Path to query audio
        track_id: Track ID to visualize matches against
        save_path: Optional path to save figure
    """
    # Get query hashes
    query_hashes = system.fingerprint_file(query_path)
    
    # Collect time pairs for this specific track
    query_times = []
    db_times = []
    
    for query_hash, query_time in query_hashes:
        if query_hash in system.database:
            for tid, db_time in system.database[query_hash]:
                if tid == track_id:
                    query_times.append(query_time)
                    db_times.append(db_time)
    
    # Create scatterplot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(query_times, db_times, alpha=0.5, s=10)
    ax.set_xlabel('Query Time (frames)')
    ax.set_ylabel('Database Time (frames)')
    ax.set_title(f'Time Pair Scatterplot - Track {track_id}')
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line for reference (perfect match would be on this line + offset)
    if query_times:
        max_time = max(max(query_times), max(db_times))
        ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.3, label='y=x reference')
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Scatterplot saved to {save_path}")
    
    plt.show()
    
    return len(query_times)


def visualize_time_difference_histogram(system, query_path, track_id, save_path=None):
    """
    Recreate Figure 2B and 3B from the paper: histogram of time differences.
    
    This shows the distribution of δt = db_time - query_time.
    A correct match shows a strong peak at the correct offset.
    
    Args:
        system: ShazamSystem instance with loaded database
        query_path: Path to query audio
        track_id: Track ID to visualize
        save_path: Optional path to save figure
    """
    # Get query hashes
    query_hashes = system.fingerprint_file(query_path)
    
    # Collect time differences for this track
    time_diffs = []
    
    for query_hash, query_time in query_hashes:
        if query_hash in system.database:
            for tid, db_time in system.database[query_hash]:
                if tid == track_id:
                    time_diff = db_time - query_time
                    time_diffs.append(time_diff)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if time_diffs:
        counts, bins, patches = ax.hist(time_diffs, bins=50, edgecolor='black')
        
        # Highlight the peak
        max_count_idx = np.argmax(counts)
        patches[max_count_idx].set_facecolor('red')
        
        peak_offset = (bins[max_count_idx] + bins[max_count_idx + 1]) / 2
        peak_count = counts[max_count_idx]
        
        ax.set_title(f'Time Difference Histogram - Track {track_id}\n'
                    f'Peak: {int(peak_offset)} frames with {int(peak_count)} matches')
    else:
        ax.set_title(f'Time Difference Histogram - Track {track_id}\nNo matches found')
    
    ax.set_xlabel('Time Difference δt = db_time - query_time (frames)')
    ax.set_ylabel('Count (Number of matching hashes)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Histogram saved to {save_path}")
    
    plt.show()
    
    return time_diffs


def analyze_database_statistics(system):
    """
    Print statistics about the database.
    
    Useful for understanding the fingerprinting results.
    """
    print("="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    
    # Basic stats
    num_tracks = len(system.track_metadata)
    num_unique_hashes = len(system.database)
    
    # Count total hash entries (some hashes appear in multiple tracks)
    total_hash_entries = sum(len(entries) for entries in system.database.values())
    
    # Hashes per track
    hashes_per_track = defaultdict(int)
    for hash_value, entries in system.database.items():
        for track_id, time_offset in entries:
            hashes_per_track[track_id] += 1
    
    avg_hashes_per_track = np.mean(list(hashes_per_track.values()))
    
    # Hash collision statistics (how many tracks share each hash)
    collision_counts = [len(entries) for entries in system.database.values()]
    
    print(f"Number of tracks: {num_tracks}")
    print(f"Number of unique hashes: {num_unique_hashes}")
    print(f"Total hash entries: {total_hash_entries}")
    print(f"Average hashes per track: {avg_hashes_per_track:.1f}")
    print(f"\nHash collision statistics:")
    print(f"  Min tracks per hash: {min(collision_counts)}")
    print(f"  Max tracks per hash: {max(collision_counts)}")
    print(f"  Avg tracks per hash: {np.mean(collision_counts):.2f}")
    print(f"  Median tracks per hash: {np.median(collision_counts):.1f}")
    
    # Storage size estimate
    # Each entry: hash (tuple of 3 ints) + track_id (int) + time_offset (int)
    # Very rough estimate
    storage_mb = (num_unique_hashes * 50) / (1024 * 1024)  # rough bytes per entry
    print(f"\nApproximate storage size: {storage_mb:.2f} MB")
    
    print("="*60)


def test_multiple_snr_levels(system, audio_path, track_id, 
                             snr_levels=[-12, -9, -6, -3, 0, 3, 6],
                             duration_sec=10):
    """
    Recreate Figure 4 from paper: Recognition rate vs SNR.
    
    Tests the same audio sample at different noise levels.
    
    Args:
        system: ShazamSystem with loaded database
        audio_path: Path to original audio file
        track_id: True track ID
        snr_levels: List of SNR levels to test (in dB)
        duration_sec: Duration of test clips
    """
    from test import create_degraded_query
    import tempfile
    import os
    
    print("Testing multiple SNR levels...")
    print(f"Track ID: {track_id}")
    print(f"SNR levels: {snr_levels}")
    
    results = []
    
    for snr in snr_levels:
        # Create temporary degraded query
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            create_degraded_query(audio_path, tmp_path, 
                                duration_sec=duration_sec,
                                noise_level_db=snr)
            
            # Query
            matches = system.query(tmp_path, top_k=1)
            
            # Check if correct
            correct = (matches and matches[0]['track_id'] == track_id)
            score = matches[0]['score'] if matches else 0
            
            results.append({
                'snr': snr,
                'correct': correct,
                'score': score
            })
            
            print(f"SNR {snr:+3d}dB: {'✓' if correct else '✗'} (score: {score})")
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    snrs = [r['snr'] for r in results]
    corrects = [r['correct'] for r in results]
    scores = [r['score'] for r in results]
    
    # Recognition rate
    ax1.plot(snrs, corrects, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Recognition (1=correct, 0=incorrect)')
    ax1.set_title(f'Recognition vs SNR ({duration_sec}s samples)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.1, 1.1])
    
    # Match scores
    ax2.plot(snrs, scores, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Match Score (number of aligned hashes)')
    ax2.set_title('Match Score vs SNR')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snr_analysis.png', dpi=150)
    print("\nPlot saved to snr_analysis.png")
    plt.show()
    
    return results


if __name__ == '__main__':
    """
    Example usage of analysis utilities.
    """
    # Load existing database
    system = ShazamSystem()
    system.load_database('shazam_database.pkl')
    
    # Print database statistics
    analyze_database_statistics(system)
    
    # Example: Analyze a specific query
    # Uncomment and modify these lines after running test_shazam.py:
    
    # query_path = 'test_queries/query_00_track_0.wav'
    # true_track_id = 0
    # 
    # # Visualize scatterplot
    # visualize_scatterplot(system, query_path, true_track_id, 
    #                      save_path='scatterplot_match.png')
    # 
    # # Visualize histogram
    # visualize_time_difference_histogram(system, query_path, true_track_id,
    #                                    save_path='histogram_match.png')
    # 
    # # Test multiple SNR levels
    # original_path = system.track_metadata[true_track_id]
    # test_multiple_snr_levels(system, original_path, true_track_id)