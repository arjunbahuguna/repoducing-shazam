# An Industrial-Strength Audio Search Algorithm (ISMIR 2003)

This is a Python implementation of the audio fingerprinting algorithm described in "An Industrial-Strength Audio Search Algorithm" by Avery Li-Chun Wang (2003) [Link](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf).

## Overview

The system consists of 3 main components:

1. Constellation Map Generation: Extract time-frequency peaks from spectrograms
2. Combinatorial Hashing: Create fingerprints by pairing nearby peaks
3. Matching & Scoring: Find matches by detecting time-aligned hash clusters

## Files

- `compute.py` - Spectrogram and peak detection functions
- `shazam.py` - Main implementation of the Shazam algorithm
- `test.py` - Script to build database and test with synthetic queries
- `utils.py` - Utilities for analyzing results and creating paper figures

## Quick Start

#### 1. Install Dependencies

```bash
pip install numpy librosa scipy matplotlib soundfile tqdm
```

#### 2. Build Database and Run Tests

```python
python src/test.py
```

This will:
- Build a fingerprint database from folders `00` and `01` in MTG-Jamendo dataset
- Save the database to `shazam_database.pkl`
- Create 10 degraded test queries (10 seconds, -6 dB SNR)
- Test matching and report accuracy

#### 3. Analyze Results

```python
python src/analyse.py
```

This will print database statistics. You can also uncomment the example code to:
- Visualize scatterplots (reproducing Figure 2A/3A from paper)
- Visualize time difference histograms (Figure 2B/3B)
- Test multiple SNR levels (Figure 4)

## Key Parameters (from Wang03 Paper)

These are set as defaults in `ShazamSystem.__init__()`:

- **Fanout (F)**: 10 - Number of target points paired with each anchor
- **Target Zone**: Time range [0, 100] frames, frequency range ±50 bins
- **Peak Detection**: 7×7 neighborhood (frequency × time)
- **Spectrogram**: FFT size 2048, hop 1024, sample rate 44100 Hz (matches our datatset)

## Paper Reference

Wang, A. (2003). "An Industrial-Strength Audio Search Algorithm." 
In Proceedings of the 4th International Conference on Music Information Retrieval (ISMIR).

The algorithm uses "combinatorially hashed time-frequency constellation analysis" to create robust audio fingerprints that survive noise, distortion, and compression. For customizing the implementation and further notes, please see CUSTOMIZATION.md
