## Customization

### Loading data from MTG-Jamnendo

You can choose the size of your subset using `test.py`:

```python
FOLDERS_TO_PROCESS = ['00', '01']  # Select only two folders from MTG-Jamendo
```

or

```python
FOLDERS_TO_PROCESS = [f'{i:02d}' for i in range(100)]  # Select all folders
```

### Adjust Test Parameters

In `test.py`:

```python
NUM_QUERIES = 10        # Number of test queries
QUERY_DURATION = 10     # Query length in seconds (paper tested 5, 10, 15)
NOISE_LEVEL = -6        # SNR in dB (paper tested -15 to +15)
```

### Change Algorithm Parameters

When creating a `ShazamSystem` instance:

```python
system = ShazamSystem(
    fanout=15,              # More pairs = more robust but slower
    target_t_end=200,       # Larger target zone
    dist_freq=10,           # Larger peak detection neighborhood
    thresh=0.02             # Higher threshold = fewer peaks
)
```

## How It Works

### 1. Fingerprinting

```
Audio File → Spectrogram → Peak Detection → Constellation Map → Hash Pairs
```

Each audio file is converted into a set of hash fingerprints:
- **Spectrogram**: Time-frequency representation of audio
- **Peaks**: Local maxima in the spectrogram (robust to noise)
- **Pairs**: Each peak (anchor) is paired with nearby peaks (targets)
- **Hash**: `(freq1, freq2, time_diff)` - 30 bits of information

### 2. Database Building

For each track in the database:
1. Generate all hash fingerprints
2. Store: `hash → [(track_id, time_offset), ...]`

### 3. Matching

For a query:
1. Generate query hashes
2. Look up each hash in database
3. For each matching hash, calculate: `δt = db_time - query_time`
4. Find tracks with many hashes at the same δt (peak in histogram)
5. Score = number of aligned hashes

**Key Insight**: If query matches a database track, their hashes will align with a consistent time offset, forming a diagonal line in the scatterplot.

## Performance Expectations (from Paper)

With clean audio (radio quality):
- **Search speed**: ~1-10 ms per query
- **Recognition rate**: >95% with 10+ second samples

With noisy audio (-6 dB SNR, GSM compressed):
- **15 second samples**: ~90% recognition
- **10 second samples**: ~75% recognition
- **5 second samples**: ~60% recognition

## Example Usage

```python
from shazam import ShazamSystem

# Create system
system = ShazamSystem()

# Build database
system.build_database(
    dataset_path='/path/to/mtg_jamendo',
    folders=['00', '01']
)

# Save database
system.save_database('my_database.pkl')

# Query a song
results = system.query('query.wav', top_k=5)

# Display results
for result in results:
    print(f"Rank {result['rank']}: Track {result['track_id']}, "
          f"Score: {result['score']}")
```

## Troubleshooting

**Out of memory**: Process fewer folders or reduce `bin_max` parameter

**Low accuracy**: 
- Increase query duration
- Reduce noise level
- Increase fanout parameter
- Check that query is actually from database

**Slow queries**: 
- Reduce fanout (fewer hashes generated)
- Reduce constellation point density (increase `thresh`)