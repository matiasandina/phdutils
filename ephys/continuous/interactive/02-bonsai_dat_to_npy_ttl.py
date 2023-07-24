import numpy as np
import subprocess
import os
import yaml
import sys
from py_console import console
from pathlib import Path
from utils import *

console.log("Choose EEG file")
eeg_file = ui_find_file(title="Choose EEG file", initialdir=os.path.expanduser("~"))
ephys_folder = os.path.dirname(eeg_file)
console.info(f"Working on {ephys_folder}")

config_file = validate_file_exists(ephys_folder, "config.yaml")
config = read_yaml(config_file)

ttl_file = validate_file_exists(ephys_folder, "ttl_in_state0")

# get sampling frequency
f_aq = config['aq_freq_hz']

ttl_file_size = os.path.getsize(ttl_file)
# for np.in8 the filesize will be the same as the total lines
total_entries = ttl_file_size
console.info(f"{ttl_file} has {total_entries} total entries")

# TODO: better split the reading so that we don't attempt to read 100 Gb of data
min_rec = 720
timepoints = f_aq * 60 * min_rec
# This is hardcoded because the I/O board has 8 channels
num_channels = 8

A = np.fromfile(ttl_file, dtype=np.int8, count=num_channels*timepoints)
# if reading all data, this will 
if A.shape[0] < num_channels*timepoints:
  # first integer division, then blowup.
  n_samples_all_channels = A.shape[0] // num_channels
  samples_to_subset = n_samples_all_channels * num_channels
  console.info(f"Reading all dataset. Reshaping and transposing into {num_channels, n_samples_all_channels}")
  # subset and reshape
  A = A[:samples_to_subset].reshape(n_samples_all_channels, num_channels).T
else:
  console.info(f"Reading {num_channels*timepoints} timepoints. Reshaping to (num_channels, timepoints)")
  A = A.reshape(timepoints, num_channels).T

A = normalize_ttl(A, method="max")

outfile = os.path.join(ephys_folder, f"{Path(ttl_file).stem}.npy")
console.success(f"Saving all {num_channels} channels")
np.save(outfile, A)
