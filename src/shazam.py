import numpy as np
import librosa
import pickle
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time

# Import functions from compute.py (your renamed plot.py)
from compute import compute_spectrogram, compute_constellation_map


class ShazamSystem:
    """
    Implementation of the Shazam audio fingerprinting algorithm based on Wang03 paper.
    
    The system works in 3 main phases:
    1. Fingerprinting: Extract time-frequency peaks and create hash tokens
    2. Database Building: Index all songs with their fingerprints
    3. Matching: Query songs against database and find matches
    """
    
    def __init__(self, 
                 Fs=44100,           # Sample rate (updated to match your files)
                 N=2048,             # FFT size
                 H=1024,             # Hop length
                 bin_max=128,        # Max frequency bin (reduces computation)
                 dist_freq=7,        # Neighborhood size for peak detection (frequency)
                 dist_time=7,        # Neighborhood size for peak detection (time)
                 thresh=0.01,        # Threshold for peak detection
                 fanout=10,          # Number of points to pair with each anchor (F in paper)
                 target_t_start=0,   # Target zone: time start offset (in frames)
                 target_t_end=100,   # Target zone: time end offset (in frames)
                 target_f_range=50): # Target zone: frequency range (in bins)
        """
        Initialize the Shazam system with parameters from the Wang03 paper.
        
        Key parameters:
        - fanout: F=10 as suggested in paper (Section 2.2)
        - target zone: Area where we look for pairs relative to anchor point
        - dist_freq/dist_time: For peak detection neighborhood
        """
        self.Fs = Fs
        self.N = N
        self.H = H
        self.bin_max = bin_max
        self.dist_freq = dist_freq
        self.dist_time = dist_time
        self.thresh = thresh
        self.fanout = fanout
        self.target_t_start = target_t_start
        self.target_t_end = target_t_end
        self.target_f_range = target_f_range
        
        # Database: maps hash -> list of (track_id, time_offset)
        self.database = defaultdict(list)
        
        # Track metadata: maps track_id -> filepath
        self.track_metadata = {}
        
    def generate_peaks(self, audio_path):
        """
        Step 1: Generate constellation map (time-frequency peaks) from audio file.
        
        From paper Section 2.1: "A time-frequency point is a candidate peak if it 
        has a higher energy content than all its neighbors in a region centered 
        around the point."
        
        Returns:
            peaks: Array of (frequency_bin, time_frame) coordinates
            Y: Spectrogram (for visualization if needed)
        """
        # Compute spectrogram
        Y = compute_spectrogram(audio_path, self.Fs, self.N, self.H, self.bin_max)
        
        # Find peaks using local maximum filtering
        Cmap = compute_constellation_map(Y, self.dist_freq, self.dist_time, self.thresh)
        
        # Extract peak coordinates: returns array of [freq_bin, time_frame] pairs
        peaks = np.argwhere(Cmap == 1)
        
        return peaks, Y
    
    def generate_hashes(self, peaks):
        """
        Step 2: Generate combinatorial hashes from peak pairs.
        
        From paper Section 2.2: "Fingerprint hashes are formed from the constellation 
        map, in which pairs of time-frequency points are combinatorially associated. 
        Anchor points are chosen, each anchor point having a target zone associated 
        with it."
        
        Each hash consists of:
        - f1: frequency of anchor point
        - f2: frequency of target point
        - dt: time difference between points
        
        The hash is: hash = (f1, f2, dt)
        We also store: time offset of the anchor point
        
        Returns:
            hashes: List of tuples (hash_value, time_offset)
                   where hash_value = (f1, f2, dt) and time_offset is anchor time
        """
        hashes = []
        
        # Sort peaks by time to make target zone selection efficient
        peaks_sorted = peaks[peaks[:, 1].argsort()]
        
        # For each peak, treat it as an anchor point
        for i, anchor in enumerate(peaks_sorted):
            anchor_freq, anchor_time = anchor
            
            # Define target zone: points that come after the anchor in time
            # and are within the frequency range
            target_zone_start_idx = i + 1
            
            # Collect points in target zone
            targets = []
            for j in range(target_zone_start_idx, len(peaks_sorted)):
                target_freq, target_time = peaks_sorted[j]
                
                # Check if target is within our target zone constraints
                time_diff = target_time - anchor_time
                freq_diff = abs(target_freq - anchor_freq)
                
                # Target zone boundaries (from paper parameters)
                if (self.target_t_start <= time_diff <= self.target_t_end and 
                    freq_diff <= self.target_f_range):
                    targets.append((target_freq, target_time))
                
                # Stop if we've gone past the time window
                if time_diff > self.target_t_end:
                    break
            
            # Limit to fanout F targets (paper suggests F=10)
            # We take the closest targets in time
            targets = targets[:self.fanout]
            
            # Create hash for each anchor-target pair
            for target_freq, target_time in targets:
                # Hash components (30 bits total in paper: 10+10+10)
                f1 = int(anchor_freq)
                f2 = int(target_freq)
                dt = int(target_time - anchor_time)
                
                # Create hash tuple (we'll use this as dictionary key)
                hash_value = (f1, f2, dt)
                
                # Time offset is the anchor point's time
                time_offset = int(anchor_time)
                
                hashes.append((hash_value, time_offset))
        
        return hashes
    
    def fingerprint_file(self, audio_path):
        """
        Complete fingerprinting pipeline for a single audio file.
        
        Returns:
            hashes: List of (hash_value, time_offset) tuples
        """
        peaks, _ = self.generate_peaks(audio_path)
        hashes = self.generate_hashes(peaks)
        return hashes
    
    def add_track_to_database(self, audio_path, track_id):
        """
        Fingerprint a track and add it to the database.
        
        From paper Section 2.2: "To create a database index, the above operation 
        is carried out on each track in a database to generate a corresponding list 
        of hashes and their associated offset times."
        
        Args:
            audio_path: Path to audio file
            track_id: Unique identifier for this track
        """
        # Generate fingerprints
        hashes = self.fingerprint_file(audio_path)
        
        # Add to database: for each hash, store (track_id, time_offset)
        for hash_value, time_offset in hashes:
            self.database[hash_value].append((track_id, time_offset))
        
        # Store metadata
        self.track_metadata[track_id] = audio_path
        
        return len(hashes)
    
    def build_database(self, dataset_path, folders=['00', '01']):
        """
        Build database from multiple audio files.
        
        Args:
            dataset_path: Root path to MTG-Jamendo dataset
            folders: List of folder names to process (default: ['00', '01'])
        """
        dataset_path = Path(dataset_path)
        track_id = 0
        
        print(f"Building database from folders: {folders}")
        
        # Collect all audio files
        audio_files = []
        for folder in folders:
            folder_path = dataset_path / folder
            if folder_path.exists():
                # Get all mp3 files in this folder
                mp3_files = list(folder_path.glob('*.mp3'))
                audio_files.extend(mp3_files)
                print(f"Found {len(mp3_files)} files in folder {folder}")
        
        print(f"Total files to process: {len(audio_files)}")
        
        # Process each file with progress bar
        for audio_file in tqdm(audio_files, desc="Fingerprinting tracks"):
            try:
                num_hashes = self.add_track_to_database(str(audio_file), track_id)
                track_id += 1
            except Exception as e:
                print(f"\nError processing {audio_file}: {e}")
                continue
        
        print(f"\nDatabase built with {track_id} tracks")
        print(f"Total hashes in database: {len(self.database)}")
        
    def match_query(self, query_hashes, threshold=5):
        """
        Step 3: Match a query against the database.
        
        From paper Section 2.3: "Each hash from the sample is used to search in 
        the database for matching hashes. For each matching hash found in the 
        database, the corresponding offset times from the beginning of the sample 
        and database files are associated into time pairs."
        
        The key insight: if a query matches a database track, their hashes will 
        align with a consistent time offset.
        
        Args:
            query_hashes: List of (hash_value, time_offset) from query
            threshold: Minimum number of matching hashes to consider a match
            
        Returns:
            matches: List of (track_id, score, time_offset) sorted by score
        """
        # Step 1: For each track, collect all matching time pairs
        # bins[track_id] = list of (db_time - query_time) differences
        bins = defaultdict(list)
        
        for query_hash, query_time in query_hashes:
            # Look up this hash in database
            if query_hash in self.database:
                # For each occurrence in database
                for track_id, db_time in self.database[query_hash]:
                    # Calculate time difference (delta_t in paper)
                    # If tracks match, this should be consistent
                    time_diff = db_time - query_time
                    bins[track_id].append(time_diff)
        
        # Step 2: For each track, find the most common time offset
        # From paper: "calculate a histogram of these δt values and scan for a peak"
        matches = []
        
        for track_id, time_diffs in bins.items():
            if len(time_diffs) < threshold:
                continue
            
            # Create histogram by counting occurrences of each time difference
            time_diff_counts = defaultdict(int)
            for td in time_diffs:
                time_diff_counts[td] += 1
            
            # Find the peak (most common time offset)
            best_offset = max(time_diff_counts.items(), key=lambda x: x[1])
            offset_value, score = best_offset
            
            matches.append((track_id, score, offset_value))
        
        # Sort by score (number of matching hashes) descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def query(self, query_audio_path, top_k=5):
        """
        Query the database with an audio sample.
        
        Args:
            query_audio_path: Path to query audio file
            top_k: Number of top matches to return
            
        Returns:
            results: List of dictionaries with match information
        """
        print(f"Querying with: {query_audio_path}")
        
        # Fingerprint the query
        start_time = time.time()
        query_hashes = self.fingerprint_file(query_audio_path)
        fingerprint_time = time.time() - start_time
        
        print(f"Generated {len(query_hashes)} hashes in {fingerprint_time:.3f}s")
        
        # Match against database
        start_time = time.time()
        matches = self.match_query(query_hashes)
        match_time = time.time() - start_time
        
        print(f"Matching completed in {match_time:.3f}s")
        
        # Format results
        results = []
        for i, (track_id, score, offset) in enumerate(matches[:top_k]):
            result = {
                'rank': i + 1,
                'track_id': track_id,
                'score': score,
                'time_offset': offset,
                'filepath': self.track_metadata.get(track_id, 'Unknown')
            }
            results.append(result)
            
        return results
    
    def save_database(self, filepath):
        """Save the database to a pickle file."""
        data = {
            'database': dict(self.database),
            'track_metadata': self.track_metadata,
            'params': {
                'Fs': self.Fs,
                'N': self.N,
                'H': self.H,
                'bin_max': self.bin_max,
                'dist_freq': self.dist_freq,
                'dist_time': self.dist_time,
                'thresh': self.thresh,
                'fanout': self.fanout,
                'target_t_start': self.target_t_start,
                'target_t_end': self.target_t_end,
                'target_f_range': self.target_f_range
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Database saved to {filepath}")
    
    def load_database(self, filepath):
        """Load the database from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.database = defaultdict(list, data['database'])
        self.track_metadata = data['track_metadata']
        
        # Restore parameters
        params = data['params']
        for key, value in params.items():
            setattr(self, key, value)
        
        print(f"Database loaded from {filepath}")
        print(f"Tracks: {len(self.track_metadata)}, Hashes: {len(self.database)}")