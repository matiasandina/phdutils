import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from halo import Halo
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from py_console import console
import os
import glob
import json
from sklearn.preprocessing import robust_scale, minmax_scale
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt
from yasa import sliding_window

# GENERAL PARAMS
win_sec = 2.5 
sampling_frequency = 100 #Hz

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


def normalize_data(self, method="robust"):
    assert method in ['robust', 'minmax'], f"Error: Scaling method must be either 'robust' (default) or 'minmax', received {method}"
    if method=="robust":
        return self.data.select(pl.all().map_batches(lambda x: pl.Series(robust_scale(x))))
    if method=="minmax":
        return self.data.select(pl.all().map_batches(lambda x: pl.Series(minmax_scale(x))))

def window_rms(signal, window_size):
    # let's make sure window_size is an int here
    window_size = int(window_size)
    num_segments = int(len(signal) // window_size)
    rms_values = np.zeros(num_segments)
    for i in range(num_segments):
        segment = signal[i * window_size: (i + 1) * window_size]
        rms_values[i] = np.sqrt(np.mean(segment ** 2))
    return rms_values

def find_valley_threshold(data):
    console.info("Finding Peaks in EMG distribution")
    # Generate kernel density estimate of the data
    kde = gaussian_kde(data)
    # Create an array of values over which to evaluate the KDE
    x_values = np.linspace(min(data), max(data), num=1000)
    # Evaluate the KDE over the range of values
    kde_values = kde(x_values)
    # Find peaks (local maxima) in the KDE to locate the modes
    peaks, _ = find_peaks(kde_values)
    # Find the x-values of the peaks for clarity
    peak_x_values = x_values[peaks]
    console.log(f"found {len(peak_x_values)} peaks at {peak_x_values}")
    # Ensure there are at least two peaks for the bimodal distribution
    if len(peaks) >= 2:
        # Find the two largest peaks
        peak_heights = kde_values[peaks]
        largest_peaks = peaks[np.argsort(peak_heights)[-2:]]
        # Ensure the indices are in the correct order
        left_peak, right_peak = sorted(largest_peaks)
        # Find the valley index between the two largest peaks
        valley_index = np.argmin(kde_values[left_peak:right_peak]) + left_peak
        threshold = x_values[valley_index]
        console.success(f"Found threshold at {threshold}")
    else:
        raise ValueError("Could not find two distinct peaks in the data.")
    return threshold, peak_x_values

def display_peak_dist(log_rms_emg, peak_x_values, threshold, filename = None):
    ax = sns.displot(log_rms_emg, kind="kde", fill=True, alpha=.8)
    # fucking matplotlib cannot handle a list
    [plt.axvline(peak, linewidth=2, color='k', linestyle = '--') for peak in peak_x_values]
    plt.axvline(threshold, color='r', linestyle = '--')
    plt.xlabel(r"log(EMG_RMS)")
    if filename is None:
        plt.show(block=False)
    else:
        # Save the figure to a file
        ax.figure.savefig(filename)
        plt.close(ax.figure)

def classify_df(data, threshold, scale):
    if scale:
        data = data.with_columns(
            pl.col("EMG1").map_batches(lambda x: pl.Series(robust_scale(x)))
        )
    data_to_save = data.with_columns(
        pl.when(pl.col("EMG1") > threshold).
        then(pl.lit(0)). # use yasa wake
        otherwise(pl.lit(2)). # use yasa NREM because we want to convert later and it will be harder
        alias("EMG1")).select(
        pl.col("EMG1"))
    return data_to_save

def classify_log_emg(log_emg, threshold):
    # use yasa wake
    # use yasa NREM because we want to convert later and it will be harder
    classified = np.where(log_emg > threshold, 0, 2)
    return classified

def process_files_in_eeg_directory(eeg_directory, overwrite, scale=True):
    results_dict = {}
    console.info(f"Processing files in {eeg_directory}")
    for file in glob.glob(os.path.join(eeg_directory, '*desc-down10_eeg.csv.gz')):
        console.success(f"Found {file}")
        # Construct the path for the plot
        session_path = os.path.dirname(eeg_directory)  # Assuming 'eeg' is directly under 'session'
        plot_dir = os.path.join(session_path, 'sleep', 'log_rms_emg')
        plot_path = os.path.join(plot_dir, os.path.basename(file).replace('desc-down10_eeg.csv.gz', 'threshold_plot.png'))
        classified_data_path = os.path.join(plot_dir, os.path.basename(file).replace('desc-down10_eeg.csv.gz', 'log_rms_classified.csv.gz'))

        # Check if specific prediction files exist
        if os.path.exists(classified_data_path) and not overwrite:
            console.warn(f"Prediction files already exist in {plot_dir} and `overwrite` is set to {overwrite}")
            console.info("Skipping folder. To recompute, re-run with `overwrite` = True")
            continue
        os.makedirs(plot_dir, exist_ok=True)
        try:
            with Halo(text=f'Processing file {file}', spinner='dots'):
                data = pl.read_csv(file)  # Process the EEG data here
            emg1 = data.select(pl.col("EMG1")).to_numpy().squeeze()
            if scale:
                console.info("Performing robust scale of emg1")
                emg1 = robust_scale(emg1)
            log_rms_emg = np.log10(window_rms(signal=emg1, window_size=win_sec * sampling_frequency))
            threshold, peak_x_values = find_valley_threshold(log_rms_emg)
            display_peak_dist(log_rms_emg, peak_x_values, threshold, plot_path)
            classified_data = classify_log_emg(log_rms_emg, threshold)
            np.savetxt(classified_data_path, classified_data.astype(int), fmt='%i', header="EMG1", comments="")
            console.success(f"Processed and saved data for {file}")
            # Append results for this file to the dictionary
            results_dict[file] = {
                'threshold': threshold,
                'peak_x_values': peak_x_values.tolist(),
            }
        except Exception as e:
            console.error(f"Failed processing {file} due to {e}")
    return results_dict

def main(root_dir, overwrite=True, scale=True):
    results_dict = {}
    # Finding all eeg directories using glob
    eeg_directories = glob.glob(os.path.join(root_dir, '*', '*', 'eeg'))
    for eeg_directory in sorted(eeg_directories):
        results = process_files_in_eeg_directory(eeg_directory, overwrite=overwrite, scale=scale)
        results_dict.update(results)
    # save results to a JSON file
    results_file = os.path.join(root_dir, 'log_rms_emg_thresholds_and_peaks.json')
    with open(results_file, 'w') as fp:
        json.dump(results_dict, fp)
    console.success("Finished processing all folders. Saved thresholds to json")


if __name__ == "__main__":
    root = "/home/matias/Experiments/eeg_24h/data"
    main(root)