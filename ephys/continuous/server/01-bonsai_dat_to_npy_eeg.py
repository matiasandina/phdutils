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
import polars as pl
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='This script reads all _eeg.bin in a folder and binds them together')

# Add arguments for ephys_folder and config_folder
parser.add_argument('--ephys_folder', help='Path to the ephys folder')
parser.add_argument('--config_folder', help='Path to the config folder')

# Parse the command-line arguments
args = parser.parse_args()

# Check if ephys_folder argument is provided, otherwise prompt the user
if args.ephys_folder:
    ephys_folder = args.ephys_folder
else:
    print("Choose any EEG file")
    eeg_file = ui_find_file(title="Choose any EEG file", initialdir=os.path.expanduser("~"))
    ephys_folder = os.path.dirname(eeg_file)

# Check if config_folder argument is provided, otherwise use ephys_folder
if args.config_folder:
    config_folder = args.config_folder
else:
    config_folder = ephys_folder

# Now you can use ephys_folder and config_folder in your script
console.info(f"Working on {ephys_folder}")
file_list = list_files(ephys_folder, pattern="_eeg.bin", full_names=True)
config = read_config(config_folder)

subject_id = config["subject_id"]
# get sampling frequency
f_aq = config['aq_freq_hz']
#total_lines = line_count(eeg_file)
#console.info(f"{eeg_file} has {total_lines} total lines")

num_channels = len(config['selected_channels'])

## look for camera saved as an integer at the start or end of the sequence
## because the sampling rate is so high, the frames will be repeated so we can use this fact
## we read 2 samples from each channel and the frames that correspond to those
#head = np.fromfile(eeg_file, dtype=np.float32,count=(num_channels + 1) * 2)
## we reshape to 2 columns for easier subtraction
#two_col = head.reshape(2, (num_channels + 1)).T
#has_camera = any(np.subtract(two_col[:, 0], two_col[:, 1]) == 0)
#
#if has_camera:
#  num_channels = num_channels + 1
#  console.info(f"camera frame present in dataset, new channel_num is {num_channels}", severe=True)

num_files = len(file_list)
##combined_data = np.zeros((num_files, num_channels, 0), dtype=np.float32)
#
## Chunk the file list
bonsai_timer_period = datetime.datetime.strptime(config["bonsai_timer_period"], "%H:%M:%S")
expected_delta_sec = datetime.datetime.timedelta(hours = bonsai_timer_period.hour, minutes= bonsai_timer_period.minute, seconds = bonsai_timer_period.second).total_seconds()
expected_delta_min = expected_delta_sec / 60
# in minutes
discontinuity_tolerance = 5
chunks = chunk_file_list(file_list, expected_delta_min, discontinuity_tolerance)

for chunk_idx, chunk in enumerate(chunks):
  console.log(f"Working on chunk {chunk_idx}/{len(chunk_idx)}")
  # Iterate over the files and combine data
  combined_data = read_stack_chunks(chunk)
  # bandpass filter data
  filtered_data = filter_data(combined_data, config, save = False)
  # Downsample
  # 'fir' is super important, all else too slow and breaking
  if config['down_freq_hz'] is None:
    console.info("No down_freq_hz in config. Exit without downsampling")
    sys.exit()
  else:
    assert config["aq_freq_hz"] > config["down_freq_hz"], f"{config['aq_freq_hz']} must be greater than {config['down_freq_hz']}"
    downsample_factor = int(config["aq_freq_hz"]/config["down_freq_hz"])
    console.info(f"Downsampling with factor {downsample_factor} from {config['aq_freq_hz']} into {config['down_freq_hz']} Hz")
    data_down = decimate(eeg_array, downsample_factor, ftype='fir')
    channel_map = create_channel_map(data_down, config)
    console.info(f"Provided channel map is {channel_map}")
    eeg_df = pl.DataFrame(data_down.T, schema = channel_map)
    # use the first timestamp of the session for each chunk
    outfilename = os.path.join(ephys_folder, f"sub-{subject_id}_ses-{session_date}_desc-down{downsample_factor}_eeg.csv.gz")
    eeg_df.write_csv(outfilename)
    console.success(f"Downsampled data written to {outfilename}")
