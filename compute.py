import numpy as np
import librosa
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt

# Some code reused from https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html

def compute_spectrogram(fn_wav, Fs=44100, N=2048, H=1024, bin_max=128, frame_max=None):
    # Fixed: librosa.load() requires sr as keyword argument
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_duration = len(x) / Fs
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    if bin_max is None:
        bin_max = X.shape[0]
    if frame_max is None:
        frame_max = X.shape[0]
    Y = np.abs(X[:bin_max, :frame_max])
    return Y

def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
    """Compute constellation map 

    Args:
        Y (np.ndarray): Spectrogram (magnitude)
        dist_freq (int): Neighborhood parameter for frequency direction (kappa) (Default value = 7)
        dist_time (int): Neighborhood parameter for time direction (tau) (Default value = 7)
        thresh (float): Threshold parameter for minimal peak magnitude (Default value = 0.01)

    Returns:
        Cmap (np.ndarray): Boolean mask for peak structure (same size as Y)
    """
    result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
    Cmap = np.logical_and(Y == result, result > thresh)
    return Cmap


def plot_constellation_map(Cmap, Y=None, xlim=None, ylim=None, title='',
                           xlabel='Time (sample)', ylabel='Frequency (bins)',
                           s=5, color='r', marker='o', figsize=(7, 3), dpi=72):
    """Args:
        Cmap: Constellation map given as boolean mask for peak structure
        Y: Spectrogram representation (Default value = None)
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for y-axis (Default value = None)
        title: Title for plot (Default value = '')
        xlabel: Label for x-axis (Default value = 'Time (sample)')
        ylabel: Label for y-axis (Default value = 'Frequency (bins)')
        s: Size of dots in scatter plot (Default value = 5)
        color: Color used for scatter plot (Default value = 'r')
        marker: Marker for peaks (Default value = 'o')
        figsize: Width, height in inches (Default value = (7, 3))
        dpi: Dots per inch (Default value = 72)

    Returns:
        fig: The created matplotlib figure
        ax: The used axes.
        im: The image plot
    """
    if Cmap.ndim > 1:
        (K, N) = Cmap.shape
    else:
        K = Cmap.shape[0]
        N = 1
    if Y is None:
        Y = np.zeros((K, N))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    im = ax.imshow(Y, origin='lower', aspect='auto', cmap='gray_r', interpolation='nearest')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    Fs = 1
    if xlim is None:
        xlim = [-0.5/Fs, (N-0.5)/Fs]
    if ylim is None:
        ylim = [-0.5/Fs, (K-0.5)/Fs]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    n, k = np.argwhere(Cmap == 1).T
    ax.scatter(k, n, color=color, s=s, marker=marker)
    plt.tight_layout()
    return fig, ax, im


def peak_pairs(peaks, fanout=10, target_t_start=0, target_t_end=100, target_f_range=50):
    """
    Generate combinatorial peak pairs for fingerprinting.
    
    From Wang03 paper Section 2.2: "Anchor points are chosen, each anchor point 
    having a target zone associated with it. Each anchor point is sequentially 
    paired with points within its target zone."
    
    Args:
        peaks (np.ndarray): Array of peak coordinates (freq_bin, time_frame)
        fanout (int): Number of target points to pair with each anchor (F in paper)
        target_t_start (int): Target zone start time offset (frames)
        target_t_end (int): Target zone end time offset (frames)
        target_f_range (int): Target zone frequency range (bins)
    
    Returns:
        pairs: List of ((f1, t1), (f2, t2)) tuples representing anchor-target pairs
    """
    pairs = []
    
    # Sort peaks by time for efficient target zone search
    peaks_sorted = peaks[peaks[:, 1].argsort()]
    
    # For each peak as anchor
    for i, anchor in enumerate(peaks_sorted):
        anchor_freq, anchor_time = anchor
        
        # Find targets in the target zone
        targets = []
        for j in range(i + 1, len(peaks_sorted)):
            target_freq, target_time = peaks_sorted[j]
            
            # Calculate differences
            time_diff = target_time - anchor_time
            freq_diff = abs(target_freq - anchor_freq)
            
            # Check if in target zone
            if (target_t_start <= time_diff <= target_t_end and 
                freq_diff <= target_f_range):
                targets.append((target_freq, target_time))
            
            # Stop if beyond time window
            if time_diff > target_t_end:
                break
        
        # Limit to fanout F closest targets
        targets = targets[:fanout]
        
        # Create pairs
        for target_freq, target_time in targets:
            pair = ((anchor_freq, anchor_time), (target_freq, target_time))
            pairs.append(pair)
    
    return pairs


def matching_function(query_hashes, database, threshold=5):
    """
    Match query hashes against database to find matching tracks.
    
    From Wang03 paper Section 2.3: "Each hash from the sample is used to search 
    in the database for matching hashes... The problem of deciding whether a match 
    has been found reduces to detecting a significant cluster of points forming a 
    diagonal line within the scatterplot."
    
    Args:
        query_hashes: List of (hash_value, time_offset) tuples from query
        database: Dict mapping hash_value -> list of (track_id, time_offset)
        threshold: Minimum number of aligned hashes to consider a match
    
    Returns:
        matches: List of (track_id, score, offset) sorted by score descending
    """
    from collections import defaultdict
    
    # Collect time pairs for each track
    # bins[track_id] = list of time differences (db_time - query_time)
    bins = defaultdict(list)
    
    for query_hash, query_time in query_hashes:
        # Look up hash in database
        if query_hash in database:
            # For each occurrence in database
            for track_id, db_time in database[query_hash]:
                # Calculate time difference
                # If tracks match, this should be consistent
                time_diff = db_time - query_time
                bins[track_id].append(time_diff)
    
    # For each track, find the peak in the time difference histogram
    matches = []
    
    for track_id, time_diffs in bins.items():
        if len(time_diffs) < threshold:
            continue
        
        # Count occurrences of each time difference
        time_diff_counts = defaultdict(int)
        for td in time_diffs:
            time_diff_counts[td] += 1
        
        # Find peak (most common offset)
        best_offset, score = max(time_diff_counts.items(), key=lambda x: x[1])
        
        matches.append((track_id, score, best_offset))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches