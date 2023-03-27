import numpy as np
import subprocess
import os
import yaml
import sys
from py_console import console
from pathlib import Path
# filtering
import mne
from mne.io import RawArray
from utils import *

console.log("Choose EEG file")
root = tk.Tk()
root.withdraw()
eeg_file = ui_find_file(title="Choose EEG file", initialdir=os.path.expanduser("~"))
ephys_folder = os.path.dirname(eeg_file)
console.info(f"Working on {ephys_folder}")
# reading the config file
config = read_config(ephys_folder)

# get sampling frequency
f_aq = config['aq_freq_hz']
total_lines = line_count(eeg_file)
console.info(f"{eeg_file} has {total_lines} total lines")

# TODO: better split the reading so that we don't attempt to read 100 Gb of data
min_rec = 720
timepoints = f_aq * 60 * min_rec
num_channels = len(config['selected_channels'])

# look for camera saved as an integer at the start or end of the sequence
# because the sampling rate is so high, the frames will be repeated so we can use this fact
# we read 2 samples from each channel and the frames that correspond to those
head = np.fromfile(eeg_file, dtype=np.float32,count=(num_channels + 1) * 2)
# we reshape to 2 columns for easier subtraction
two_col = head.reshape(2, (num_channels + 1)).T
has_camera = any(np.subtract(two_col[:, 0], two_col[:, 1]) == 0)

if has_camera:
  num_channels = num_channels + 1
  console.info(f"camera frame present in dataset, new channel_num is {num_channels}", severe=True)

# data is stored as np.float32
eeg_array = np.fromfile(eeg_file, dtype=np.float32,count=num_channels*timepoints)
# if reading all data, this will 
if eeg_array.shape[0] < num_channels*timepoints:
  # first integer division, then blowup.
  n_samples_all_channels = eeg_array.shape[0] // num_channels
  samples_to_subset = n_samples_all_channels * num_channels
  console.info(f"Reading all dataset. Reshaping and transposing into {num_channels, n_samples_all_channels}")
  # subset and reshape
  eeg_array = eeg_array[:samples_to_subset].reshape(n_samples_all_channels, num_channels).T
else:
  console.info(f"Reading {num_channels*timepoints} timepoints. Reshaping to (num_channels, timepoints)")
  eeg_array = eeg_array.reshape(timepoints, num_channels).T


# Filtering
channel_map = create_channel_map(eeg_array, config)
# create filtered and raw array
bandpass_freqs = config['bandpass']['eeg']
emg_bandpass = config['bandpass']['emg']

if has_camera:
  cam_index = np.where(['cam' in value for value in list(channel_map.values())])
  cam_array = eeg_array[cam_index[0], :]
  eeg_array = np.delete(eeg_array, cam_index, axis=0)
  # save camera
  suffix = f"_camframes.npy"
  outfile = os.path.join(ephys_folder, f"{Path(eeg_file).stem}{suffix}")
  console.success(f"Saving camera frames as {outfile}")
  np.save(outfile, cam_array)

emg_channels = find_channels(config, "EMG")
if emg_channels is None:
  console.error("Indices for EMG Channels not found in channel map before filtering.\nCheck your data!\nExiting program.", severe=True)
else:
  console.success(f"Found EMG channels at idx {emg_channels}. Not filtering those")

# we will not filter emg_channels
eeg_filter_idx = [i for i in range(eeg_array.shape[0]) if i not in emg_channels]

# filter eegs
filtered_data = mne.filter.filter_data(eeg_array.astype('float64'), 
                                  sfreq = f_aq, 
                                  l_freq = min(bandpass_freqs), 
                                  h_freq = max(bandpass_freqs), 
                                  picks = eeg_filter_idx,
                                  verbose=0, n_jobs=2)

# filter emgs
filtered_data = mne.filter.filter_data(filtered_data, 
                                  sfreq = f_aq, 
                                  l_freq = min(emg_bandpass), 
                                  h_freq = max(emg_bandpass), 
                                  picks = emg_channels,
                                  verbose=0, n_jobs=2)


suffix = f"_desc-filt-{min(bandpass_freqs)}-{max(bandpass_freqs)}_eeg.npy"
outfile = os.path.join(ephys_folder, f"{Path(eeg_file).stem}{suffix}")
console.success(f"Saving eegdata as {outfile}")
np.save(outfile, filtered_data.astype('float32'))
