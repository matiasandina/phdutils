import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch

win_sec = 2.5
sf = 100
window = win_sec * sf

eeg_data = pl.read_csv('MLA169/2024-04-07/eeg/sub-MLA169_ses-20240407T003422_desc-down10_eeg.csv.gz', columns = ['EEG8'])
# this will give nrows the sleep prediction will have and remainder rows 
sleep_prediction_samples, resto = divmod(eeg_data.shape[0], window)
log_rms_data = pl.read_csv('MLA169/2024-04-07/sleep/log_rms_emg/sub-MLA169_ses-20240407T003422_log_rms_classified.csv.gz', columns = 0)

def clean_short_chunks_pl(data, column_name, min_length=5):
    # Detect changes and assign a group ID to each segment
    data = data.with_columns(
        (data[column_name] != data[column_name].shift()).cumsum().alias("group_id")
    )
    # Count the number of occurrences per group
    group_sizes = data.groupby("group_id").agg(pl.count().alias("count"))
    # Merge back group sizes to filter out small groups
    data = data.join(group_sizes, on="group_id")
    # Mark short chunks as None and keep the original group info for verification
    data = data.with_columns(
        pl.when(data["count"] < min_length)
        .then(None)
        .otherwise(data[column_name])
        .alias(column_name),
        data["group_id"].alias("original_group_id"),  # Preserve original group_id
        data["count"].alias("original_count")         # Preserve original count
    )
    # Forward fill to replace None values
    data = data.fill_null(strategy="forward")
    # Recalculate groups and their sizes after forward fill
    data = data.with_columns(
        (data[column_name] != data[column_name].shift()).cumsum().alias("new_group_id")
    )
    new_group_sizes = data.groupby("new_group_id").agg(pl.count().alias("new_count"))
    data = data.join(new_group_sizes, on="new_group_id")
    return data


def extract_segments_and_compute_spectra(eeg_data, cleaned_log_rms, column, fs=100, win_sec = 2.5):
    eeg_data = eeg_data.to_numpy().squeeze()
    # Find continuous chunks in the state data
    # Assume `cleaned_log_rms` has a 'state' and 'new_group_id' columns after cleaning
    group_details = cleaned_log_rms.with_row_index().groupby('new_group_id', maintain_order  = True).agg([
        pl.first(column).alias(column),
        pl.min('index').alias('start_index'),
        pl.max('index').alias('end_index')
    ])
    # Results storage
    spectra_results = []
    # Loop through each group, extract EEG data, compute spectrum
    for row in group_details.iter_rows(named = True):
        start = int(row['start_index'] * fs * win_sec)  # Convert to EEG data index
        end = int(row['end_index']  * fs * win_sec + 1) # Convert to EEG data index
        segment = eeg_data[start:end]
        # Compute Welch's power spectrum
        # scaling{ ‘density’, ‘spectrum’ }, optional
        # Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the squared magnitude spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’
        frequencies, power = welch(segment, fs=fs, nperseg=256, scaling="spectrum")
        # Store results
        spectra_results.append({
            'state': row[column],
            'frequencies': frequencies,
            'power': power
        })
    return spectra_results


def plot_state_spectra(spectra_results):
    # Organize data by state
    spectra_by_state = {}
    for result in spectra_results:
        state = result['state']
        if state not in spectra_by_state:
            spectra_by_state[state] = []
        spectra_by_state[state].append((result['frequencies'], result['power']))
    # Define a color palette
    colors = sns.color_palette("pastel", len(spectra_by_state))
    # Number of plots
    num_states = len(spectra_by_state)
    fig, axes = plt.subplots(num_states, 1, figsize=(10, 5 * num_states), sharex=True)
    if num_states == 1:  # Handle case of single subplot for consistency in indexing axes
        axes = [axes]
    # Plot each state's spectra
    for ax, (state, spectra) in zip(axes, spectra_by_state.items()):
        color = colors.pop(0)
        for frequencies, power in spectra:
            ax.plot(frequencies, power, color=color, alpha=0.2)  # Adjust alpha for transparency
        ax.set_title(f"Power Spectrum for State {state}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power uV^2")
    plt.tight_layout()
    plt.show()

import numpy as np

def normalize_and_average_spectra_by_state(spectra_results):
    spectra_by_state = {}
    # Organize spectra by state
    for result in spectra_results:
        state = result['state']
        if state not in spectra_by_state:
            spectra_by_state[state] = []
        total_power = np.sum(result['power'])
        normalized_power = result['power'] / total_power
        spectra_by_state[state].append(normalized_power)
    # Average the normalized spectra for each state
    average_spectra = {}
    for state, normalized_powers in spectra_by_state.items():
        # Stack all normalized power spectra for this state and calculate mean along the first axis
        mean_spectrum = np.mean(np.vstack(normalized_powers), axis=0)
        average_spectra[state] = mean_spectrum
    return average_spectra

# Plot the average spectra
def plot_average_spectra(average_spectra, frequencies):
    plt.figure(figsize=(10, 6))
    for state, power in average_spectra.items():
        plt.plot(frequencies, power, label=f'Normalized Average Spectrum for State {state}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power uV^2')
    plt.title('Normalized Average Power Spectrum by State')
    plt.legend()
    plt.show()




# fill the small chunks
clean_sleep = clean_short_chunks_pl(log_rms_data, "EMG1")
# calculate spectrums for each chunk
spectra = extract_segments_and_compute_spectra(eeg_data, clean_sleep, 'EMG1')
# plot spectrums for each chunk
plot_state_spectra(spectra)

# Assuming spectra_results is already populated
average_spectra = normalize_and_average_spectra_by_state(spectra)
# Call the plot function with frequencies from the Welch computation
# Assume 'frequencies' is available from the previous Welch computations
plot_average_spectra(average_spectra, spectra[0]['frequencies'])

