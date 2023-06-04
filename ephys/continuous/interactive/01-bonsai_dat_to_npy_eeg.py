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
from rlist_files import list_files

console.log("Choose EEG file")
eeg_file = ui_find_file(title="Choose EEG file", initialdir=os.path.expanduser("~"))
ephys_folder = os.path.dirname(eeg_file)
console.info(f"Working on {ephys_folder}")
file_list = list_files(ephys_folder, pattern = "eegdata*.bin", full_names = True)
# reading the config file
config = read_config(ephys_folder)

# get sampling frequency
f_aq = config['aq_freq_hz']
#total_lines = line_count(eeg_file)
#console.info(f"{eeg_file} has {total_lines} total lines")

# TODO: better split the reading so that we don't attempt to read 100 Gb of data
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

num_files = len(file_list)
#combined_data = np.zeros((num_files, num_channels, 0), dtype=np.float32)

# Iterate over the files
for file_idx, file in enumerate(file_list):
  console.log(f"Read data from file {file} ({file_idx+1}/{num_files})")
  # data is stored as np.float32
  eeg_array = np.fromfile(file, dtype=np.float32)
  # first integer division, then blowup.
  n_samples_all_channels = eeg_array.shape[0] // num_channels
  console.info(f"Reading all dataset. Reshaping and transposing into {num_channels, n_samples_all_channels}")
  # subset and reshape
  eeg_array = eeg_array.reshape(n_samples_all_channels, num_channels).T

  # Filtering
  channel_map = create_channel_map(eeg_array, config)
  # create filtered and raw array
  bandpass_freqs = config['bandpass']['eeg']
  emg_bandpass = config['bandpass']['emg']

  # remove cam stamp if present
  if has_camera:
    cam_index = np.where(['cam' in value for value in list(channel_map.values())])
    cam_array = eeg_array[cam_index[0], :]
    eeg_array = np.delete(eeg_array, cam_index, axis=0)
    # save camera
    suffix = f"_camframes.npy"
    outfile = os.path.join(ephys_folder, f"{Path(eeg_file).stem}{suffix}")
    console.success(f"Saving camera frames as {outfile}")
    np.save(outfile, cam_array)

  if file_idx == 0:
      # For the first file, initialize combined_data with the shape of eeg_array
      combined_data = np.expand_dims(eeg_array, axis=0)
  else: 
      # For subsequent files, expand dimensions of eeg_array before concatenating with combined_data
      combined_data = np.concatenate((combined_data, np.expand_dims(eeg_array, axis=0)), axis=0)

  console.log(f"Combined Data has the shape (files, channels, timepoints): {combined_data.shape}")


console.info("Stacking data horizontally")
combined_data = np.hstack(combined_data)
emg_channels = find_channels(config, "EMG")

if emg_channels is None:
  console.error("Indices for EMG Channels not found in channel map before filtering.\nCheck your data!\nExiting program.", severe=True)
  sys.exit(0)
else:
  console.success(f"Found EMG channels at idx {emg_channels}. Not filtering those")

# we will not filter emg_channels
eeg_filter_idx = [i for i in range(eeg_array.shape[0]) if i not in emg_channels]

# filter eegs
filtered_data = mne.filter.filter_data(combined_data.astype('float64'), 
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
