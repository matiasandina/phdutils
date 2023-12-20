from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt
import polars as pl
import numpy as np
def compute_hilbert(electrode, sampling_frequency):
    # Compute the Hilbert transform for each band for the entire dataset
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "sigma": (8, 15), #Kjaerby2022
    }
    # DataFrame to store the amplitude envelope of each band
    envelopes = pl.DataFrame()
    filtered_df = pl.DataFrame()
    for band, (low, high) in bands.items():
        # Apply bandpass filter
        sos = butter(10, [low, high], btype='band', fs=sampling_frequency, output='sos')
        #print(pl.DataFrame(sos))
        filtered = sosfiltfilt(sos, electrode)
        #print(filtered[:5])
        # Apply Hilbert transform to get the envelope (i.e., the amplitude) of the signal
        analytic_signal = hilbert(filtered)
        amplitude_envelope = np.abs(analytic_signal)
        # Store the envelope in the DataFrame
        envelopes = envelopes.with_columns(pl.Series(band, amplitude_envelope))
        # Store the filtered signals
        filtered_df = filtered_df.with_columns(pl.Series(band, filtered))
    return envelopes, filtered_df

