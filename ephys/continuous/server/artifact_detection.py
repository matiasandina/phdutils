from halo import Halo
import sys
import time
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import polars as pl
import os
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import minmax_scale, robust_scale
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt
from lspopt import spectrogram_lspopt
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def window_rms(signal, window_size = 250):
    # window_size here is win_sec (2.5) * sampling frequency (100)
    # let's make sure window_size is an int here
    window_size = int(window_size)
    num_segments = int(len(signal) // window_size)
    rms_values = np.zeros(num_segments)
    for i in range(num_segments):
        segment = signal[i * window_size: (i + 1) * window_size]
        rms_values[i] = np.sqrt(np.mean(segment ** 2))
    return rms_values

def normalize_data(data, method="robust"):
    assert method in ['robust', 'minmax'], f"Error: Scaling method must be either 'robust' (default) or 'minmax', received {method}"
    if method=="robust":
        return data.select(pl.all().map(lambda x: pl.Series(robust_scale(x))))
    if method=="minmax":
        return data.select(pl.all().map(lambda x: pl.Series(minmax_scale(x))))

def compute_hilbert(selected_electrode, sampling_frequency=100):
    # Compute the Hilbert transform for each band for the entire dataset
    bands = {
        'total': (0.5, sampling_frequency/2 - 0.00001),
        "delta": (0.5, 4),
        "theta": (4, 8),
        #"alpha": (8, 13),
        #"beta": (13, 30),
        #"gamma": (30, 100),
    }
    # DataFrame to store the amplitude envelope of each band
    envelopes = pl.DataFrame()
    for band, (low, high) in bands.items():
        # Apply bandpass filter
        sos = butter(10, [low, high], btype='band', fs=sampling_frequency, output='sos')
        filtered = sosfiltfilt(sos, selected_electrode)
        # Apply Hilbert transform to get the envelope (i.e., the amplitude) of the signal
        analytic_signal = hilbert(filtered)
        amplitude_envelope = np.abs(analytic_signal)
        # Store the envelope in the DataFrame
        envelopes = envelopes.with_columns(pl.Series(band, amplitude_envelope))
    return envelopes

def clip_quantiles(df, column, upper_quantile=0.999, lower_quantile=0.001):
    # Compute the upper and lower bounds for clipping
    upper_bound = df[column].quantile(upper_quantile)
    lower_bound = df[column].quantile(lower_quantile)
    # Clip the values beyond the upper and lower bounds
    df = df.with_columns(
        pl.when(df[column] > upper_bound)
        .then(upper_bound)
        .when(df[column] < lower_bound)
        .then(lower_bound)
        .otherwise(df[column])
        .alias(column)
    )
    return df



def power_plots(envelopes):
    plt.figure("Power Plots")  # Create a new figure
    for i, band in enumerate(envelopes.columns):
        # Select the amplitude envelope for the current band
        amplitude_envelope = envelopes[band].to_numpy()
        # Plot the envelope
        plt.plot(amplitude_envelope, label=f"{band} band")
    plt.legend()  # Add a legend
    plt.show(block=False)  # Display the plot

def pretty_time_label(seconds):
    # this function ensures that we have a format HH:MM:SS.f with only one decimal place
    stamp = datetime.datetime(1,1,1) + timedelta(seconds = seconds)
    return stamp.strftime('%H:%M:%S.%f')[:-5]

# downsampling for x axis ticks
def downsample(array, sampling_frequency, factor= 1):
    factor = sampling_frequency * factor
    return [val for i, val in enumerate(array) if divmod(i, factor)[1] == 0]

def sampled_scatter(data, n):
    # angeles@mit.edu
    import seaborn as sns
    sample = data.sample(n = n, seed = 2).to_pandas()
    sns.scatterplot(data=sample, x="delta", y="total", alpha = 0.2)
    plt.show(block=False)
    


def eeg_plots(data, selected_electrode_name, sampling_frequency=100, win_sec=2.5, plot_from=0, plot_to=None):
    # calculate x axis stuff here
    sample_axis = np.arange(0, data.shape[0], 1)
    if plot_to is None:
        plot_to = max(sample_axis)
    # Compute buffer
    plot_from_buffer = max(int(round(plot_from - win_sec * sampling_frequency)), 0)
    # Compute x-axis for plot: use plot_from_buffer here
    plot_x_axis = sample_axis[plot_from_buffer:plot_to]
    # Conversion from sample domain to time domain
    plot_x_axis_time = np.array(plot_x_axis) / sampling_frequency
    # Create tick labels
    x_tick_labels = np.arange(0, plot_x_axis_time[-1], win_sec)
    x_tick_labels_str = [pretty_time_label(seconds = i) for i in x_tick_labels]
    # Create the tick positions for these labels in the sample domain
    #x_tick_positions = x_tick_labels * sampling_frequency
    x_tick_positions = [int(round(label * sampling_frequency)) for label in x_tick_labels]
    fig, axs = plt.subplots(2, sharex=True)  # Create a new figure with two subplots
    # Then plot the data
    for i, col in enumerate(data.columns):
        y_values = data[col].to_numpy()[plot_from_buffer:plot_to]
        axs[0].plot(plot_x_axis, y_values + i, label = f"Channel index {i}")
    axs[0].legend()  # Add a legend to the first subplot
    #axs[0].set_xlabel('Time')  # Set x-label
    axs[0].set_ylabel('Electrical signal (uV)')  # Set y-label
    axs[0].set_xticks(x_tick_positions)  # Set x-ticks
    axs[0].set_xticklabels(x_tick_labels_str)  # Set x-tick labels
    # Selected electrode
    axs[1].plot(plot_x_axis, data[selected_electrode_name].to_numpy()[plot_from_buffer:plot_to], 'k', linewidth=2)
    axs[1].set_xlabel('Time')  # Set x-label
    axs[1].set_ylabel('Electrical signal (uV)')  # Set y-label
    axs[1].set_xticks(x_tick_positions)  # Set x-ticks
    axs[1].set_xticklabels(x_tick_labels_str)  # Set x-tick labels
    plt.show(block=False)  # Display the plot



def scale_inspect_scatter(data, selected_electrode):
    assert data.shape[0] < 50000, "Please sample your data 50K points is painful"
    sub = data[[selected_electrode]].to_pandas()  # Convert to pandas DataFrame
    sub["robust_scaled"] = robust_scale(sub[selected_electrode])  # Apply robust_scale
    #return pl.from_pandas(sub)  # Convert back to polars DataFrame
    sns.scatterplot(sub, x=selected_electrode, y="robust_scaled", alpha =0.5)
    plt.show(block = False)




def clip_quantiles_startswith(df, prefix="EEG", upper_quantile=0.999, lower_quantile=0.001):
    # Identify columns that start with the specified prefix
    columns_to_clip = [col for col in df.columns if col.startswith(prefix)]
    
    # Iterate over these columns and apply the clipping
    for column in columns_to_clip:
        upper_bound = df[column].quantile(upper_quantile)
        lower_bound = df[column].quantile(lower_quantile)
        df = df.with_columns(
            pl.when(df[column] > upper_bound)
            .then(upper_bound)
            .when(df[column] < lower_bound)
            .then(lower_bound)
            .otherwise(df[column])
            .alias(column)
        )
    return df


def construct_electrode_df(data, clipped, selected_electrode):
    # Extract the selected electrode column from the original data
    original_electrode = data.select(selected_electrode)
    
    # Extract the clipped version of the electrode column
    clipped_electrode = clipped.select(selected_electrode)
    
    # Apply robust scaling to the original electrode column
    # Note: robust_scale expects a 2D array, so we use to_numpy to convert and then flatten the result back to 1D
    scaled_electrode = robust_scale(original_electrode.to_numpy().reshape(-1, 1)).flatten()
    
    # Construct a new DataFrame with the original, clipped, and robustly scaled versions of the electrode
    sub_clip = pl.DataFrame({
        "original": original_electrode[selected_electrode],
        "clipped": clipped_electrode[selected_electrode],
        "robust_scaled": scaled_electrode
    })
    
    return sub_clip


def continuous_sample(df, n):
    """
    Selects a continuous sample of n rows from a Polars DataFrame.

    Parameters:
    - df: The Polars DataFrame from which to sample.
    - n: The number of continuous rows to sample.

    Returns:
    - A continuous sample of n rows from df as a Polars DataFrame.
    """
    max_start_index = len(df) - n  # Calculate the maximum start index
    if max_start_index <= 0:
        # If n is greater than or equal to the total number of rows, return the entire DataFrame
        return df
    start_index = np.random.randint(0, max_start_index)  # Randomly choose a start index
    return df.slice(start_index, n)



def find_mismatches(data, clipped, selected_electrode):
    """
    Finds indices where original and clipped values differ.
    
    Parameters:
    - data: DataFrame with original data.
    - clipped: DataFrame with clipped data.
    - selected_electrode: The column name of the electrode to analyze.
    
    Returns:
    - A numpy array of mismatch indices.
    """
    mismatches = (data[selected_electrode] != clipped[selected_electrode]).to_numpy().nonzero()[0]
    return mismatches

def find_continuous_mismatches(data, clipped, selected_electrode, n = 3):
    """
    Finds start indices of significant continuous mismatches between original and clipped values,
    disregarding blocks of mismatches shorter than `n`.
    
    Parameters:
    - data: DataFrame with original data.
    - clipped: DataFrame with clipped data.
    - selected_electrode: The column name of the electrode to analyze.
    - n: Minimum length of continuous mismatches of interest.
    
    Returns:
    - A list of start indices for continuous mismatches of interest.
    - A dictionary summarizing the distribution of all found mismatch lengths.
    """
    # Find mismatch indices
    mismatches = (data[selected_electrode] != clipped[selected_electrode]).to_numpy().nonzero()[0]
    
    if len(mismatches) == 0:
        return [], {}
    
    # Find consecutive sequences using np.diff
    diff = np.diff(mismatches)
    breaks = np.where(diff > 1)[0]  # Identify where the sequence breaks (non-consecutive)
    
    # Start indices of continuous blocks
    start_indices = np.insert(mismatches[breaks + 1], 0, mismatches[0])
    # End indices of continuous blocks
    end_indices = np.append(mismatches[breaks], mismatches[-1])
    
    # Lengths of continuous blocks
    lengths = end_indices - start_indices + 1
    
    # Filter blocks based on the length
    significant_starts = start_indices[lengths >= n]
    
    # Analyzing the distribution of mismatch lengths
    unique_lengths, counts = np.unique(lengths, return_counts=True)
    distribution = dict(zip(unique_lengths, counts))
    
    return significant_starts.tolist(), distribution

def plot_mismatch_windows(data, clipped, mismatches, sf=100, win_sec=10, selected_electrode="EEG9", samples=20):
    """
    Plots sampled `win_sec` windows around mismatch indices.
    
    Parameters:
    - data: DataFrame with original data.
    - clipped: DataFrame with clipped data.
    - mismatches: Array of mismatch indices.
    - sf: Sampling frequency in Hz.
    - win_sec: Window size in seconds for plotting.
    - selected_electrode: The column name of the electrode to analyze.
    - samples: Number of mismatch samples to plot.
    """
    if len(mismatches) == 0:
        print("No mismatches found.")
        return
    
    # If there are fewer mismatches than requested samples, reduce the sample size
    samples = min(samples, len(mismatches))
    
    # Randomly select sample mismatches for plotting
    sampled_indices = np.random.choice(mismatches, size=samples, replace=False)
    
    # Calculate half the window size in data points
    half_window_points = int(win_sec * sf / 2)
    
    # Setup subplot dimensions
    cols = min(5, samples)
    rows = (samples + cols - 1) // cols  # Ensure enough subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    
    # Flatten axes array for easy iteration if it's multidimensional
    if samples > 1:
        axes = axes.flatten()
    
    for i, index in enumerate(sampled_indices):
        # Calculate start and end of the window
        start = max(0, index - half_window_points)
        end = min(len(data), index + half_window_points)
        
        # Extract the continuous sample for original and clipped data
        original_sample = data[selected_electrode][start:end].to_numpy()
        clipped_sample = clipped[selected_electrode][start:end].to_numpy()
        
        # Generate a time axis for the plot
        time_axis = np.linspace(-win_sec / 2, win_sec / 2, len(original_sample))
        
        # Plot
        ax = axes[i] if samples > 1 else axes
        ax.plot(time_axis, original_sample, label='Original', alpha=0.75)
        ax.plot(time_axis, clipped_sample, label='Clipped', alpha=0.75)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        time_sec = "{:0>8}".format(str(timedelta(seconds=index/sf)))
        ax.set_title(f"Mismatch around index {index}.\nTime {time_sec}")
        if i // cols == rows - 1:  # Check if the subplot is in the bottom row
            ax.set_xlabel('Time around mismatch (seconds)')
        else:
            ax.set_xlabel('')
        ax.set_ylabel('Amplitude')
        if i == 0:  # Add legend only to the first plot to avoid clutter
            ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()



def find_artifacts_by_power(envelopes, threshold_factor=3, min_length = 3):
    """
    Identifies continuous regions where the amplitude envelope is considered too high, likely indicating artifacts.
    
    Parameters:
    - envelopes: DataFrame containing the amplitude envelopes of different frequency bands.
    - threshold_factor: The factor to multiply by the standard deviation to set the threshold.
    - min_length: The minimum length of consecutive points above the threshold to consider as an artifact.
    
    Returns:
    - A list of start indices for continuous stretches above the threshold of interest.
    - A dictionary summarizing the distribution of continuous stretch lengths.
    """
    artifact_starts = {}
    artifact_distribution = {}
    for band in envelopes.columns:
        # Find where the amplitude envelope exceeds the threshold
        mismatches = envelopes.with_columns(
            (pl.col(band).mean() + threshold_factor * pl.col(band).std()).alias('threshold')
            ).with_columns(
                (pl.col('total') > pl.col('threshold')).alias('flag')
            ).select(pl.col('flag')).to_series().to_numpy().nonzero()
        # unlist it
        mismatches = mismatches[0]        
        if len(mismatches) == 0:
            artifact_starts[band] = []
            artifact_distribution[band] = {}
            continue
        # Find consecutive sequences
        diff = np.diff(mismatches)
        breaks = np.where(diff > 1)[0]
        # Start indices of continuous blocks
        start_indices = np.insert(mismatches[breaks + 1], 0, mismatches[0])
        # End indices of continuous blocks
        end_indices = np.append(mismatches[breaks], mismatches[-1])
        # Lengths of continuous blocks
        lengths = end_indices - start_indices + 1
        # Filter blocks based on the length
        significant_starts = start_indices[lengths >= min_length]
        # Analyzing the distribution of mismatch lengths
        unique_lengths, counts = np.unique(lengths, return_counts=True)
        distribution = dict(zip(unique_lengths, counts))
        artifact_starts[band] = significant_starts.tolist()
        artifact_distribution[band] = distribution
    return artifact_starts, artifact_distribution

#################################################################
#           Run Script Stuff Below Here                         #
#################################################################

eeg_file = "/home/matias/Experiments/eeg_24h/data/MLA158/2024-03-18/eeg/sub-MLA158_ses-20240318T003030_desc-down10_eeg.csv.gz"
with Halo(text='Loading data...', spinner='dots'):
    data = pl.read_csv(eeg_file)  # Local variable to hold the data

# This is too much, do not run this without plot_from and plot_to values
# eeg_plots(data, "EEG8", plot_from=0, plot_to=1000)

# full power envelopes
envelopes = compute_hilbert(data["EEG9"])

sampled_scatter(envelopes, n = 10000)
plt.show(block=True)

sns.histplot(envelopes, x="total")
plt.show(block=False)

scale_inspect_scatter(data.sample(n=20000), "EEG9")
plt.show()

clipped = clip_quantiles_startswith(data,
                                    lower_quantile=0.01,
                                    upper_quantile=0.99)
start_idx, distribution = find_continuous_mismatches(data,clipped, "EEG9", n = 3)

sub = construct_electrode_df(data, clipped, "EEG9")
sub = continuous_sample(sub, n=100000)
sns.scatterplot(sub, x="original", y="clipped", alpha=0.5)
plt.show()
plot_mismatch_windows(data, clipped, mismatches=start_idx)


# Let's see this in power

artifact_starts, artifact_distribution = find_artifacts_by_power(
    envelopes[['total']], threshold_factor=4)

# artifacts based on power
plot_mismatch_windows(data, clipped, mismatches = artifact_starts['total'])



# using rolling std method from yasa
# there are different flavors of this
# check docs here
# https://github.com/raphaelvallat/yasa/blob/master/notebooks/13_artifact_rejection.ipynb

import yasa

def local_std_artifacts(data, sf, win_sec, threshold=3, visualize = False):
    # let's match win_sec so we can use the indices to go at specific predictions?
    art_std, zscores_std = yasa.art_detect(
        data.select(
            pl.selectors.starts_with('EEG')
            ).to_numpy().transpose(), 
        sf=sf, window=win_sec, 
        method='std', 
        n_chan_reject=3, # at least 3 channels for rejection
        threshold=threshold, verbose='info')
    if visualize:
        avg_zscores = zscores_std.mean(-1)
        sns.displot(avg_zscores)
        plt.title('Histogram of z-scores')
        plt.xlabel('Z-scores')
        plt.ylabel('Density')
        plt.axvline(threshold, color='r', label='Threshold')
        plt.axvline(-threshold, color='r')
        plt.legend(frameon=False)
        plt.show(block=False)
    if sum(art_std) > 0:
        artifact_idx = np.where(art_std)[0]
        return artifact_idx * sf * win_sec
    else:
        print("No Artifacts found!!")
        return None




# The inner workings of the method are as follows 
yasa.sliding_window(data.select(
            pl.selectors.starts_with('EEG')
            ).to_numpy().transpose(), 100, 2.5)[1]
# gives you an array [0] with epoch times (np.arange(0, len, win_sec))
# gives you the 3D array of (n_epochs, n_chan, n_samples)

# Epoch the data (n_epochs, n_chan, n_samples)
_, epochs = yasa.sliding_window(data, sf, window=win_sec)
n_epochs = epochs.shape[0]

# Calculate log-transformed standard dev in each epoch
# We add 1 to avoid log warning id std is zero (e.g. flat line)
# (n_epochs, n_chan)
std_epochs = np.log(np.nanstd(epochs, axis=-1) + 1)
# Create empty zscores output (n_epochs, n_chan)
zscores = np.zeros((n_epochs, n_chan), dtype="float") * np.nan
for stage in include:
    where_stage = np.where(hypno_win == stage)[0]
    # At least 30 epochs are required to calculate z-scores
    # which amounts to 2.5 minutes when using 5-seconds window
    if where_stage.size < 30:
        if hypno is not None:
            # Only show warnig if user actually pass an hypnogram
            logger.warning(
                f"At least 30 epochs are required to "
                f"calculate z-score. Skipping "
                f"stage {stage}"
            )
        continue
    # Calculate z-scores of STD for each channel x stage
    c_mean = np.nanmean(std_epochs[where_stage], axis=0, keepdims=True)
    c_std = np.nanstd(std_epochs[where_stage], axis=0, keepdims=True)
    zs = (std_epochs[where_stage] - c_mean) / c_std
    # Any epoch with at least X channel above or below threshold
    n_chan_supra = (np.abs(zs) > threshold).sum(axis=1)  # >
    art = (n_chan_supra >= n_chan_reject).astype(int)  # >= !
    if hypno is not None:
        # Only shows if user actually pass an hypnogram
        perc_reject = 100 * (art.sum() / art.size)
        text = (
            f"Stage {stage}: {art.sum()} / {art.size} "
            f"epochs rejected ({perc_reject:.2f}%)"
        )
        logger.info(text)
    # Append to global vector
    epoch_is_art[where_stage] = art
    zscores[where_stage, :] = zs