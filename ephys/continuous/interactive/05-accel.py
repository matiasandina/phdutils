import numpy as np
import subprocess
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
import yaml
import sys
from py_console import console
from pathlib import Path
# filtering
import mne
from mne.io import RawArray
from utils import *

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

def read_yaml(filename):
  with open(filename, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
  return cfg

console.log("Choose acceleration file")
root = tk.Tk()
root.withdraw()
acc_file = ui_find_file(title="Choose accel file", initialdir=os.path.expanduser("~"))
ephys_folder = os.path.dirname(acc_file)
console.info(f"Working on {ephys_folder}")

# check for video available
config_file = [os.path.join(ephys_folder, file_name) for file_name in os.listdir(ephys_folder) if 'config.yaml' in file_name]
if not config_file:
  console.error(f"config.yaml not found in {ephys_folder}", severe=True)
  sys.exit()
else:
  # unlist using first element, assume only one match...
  config_file = config_file[0]

config = read_yaml(config_file)

# get sampling frequency
f_aq = config['aq_freq_hz'] // 4 # accelerometer will always be 1/4 of eeg aq freq
total_lines = line_count(acc_file)
console.info(f"{acc_file} has {total_lines} total lines")

# TODO: better split the reading so that we don't attempt to read 100 Gb of data
# TODO: it's not clear how the data would work if we have more than one animal here
min_rec = 720
timepoints = f_aq * 60 * min_rec
num_channels = 3


# data is stored as np.float32
acc_array = np.fromfile(acc_file, dtype=np.float32,count=num_channels*timepoints)
# if reading all data, this will 
if acc_array.shape[0] < num_channels*timepoints:
  # first integer division, then blowup.
  n_samples_all_channels = acc_array.shape[0] // num_channels
  samples_to_subset = n_samples_all_channels * num_channels
  console.info(f"Reading all dataset. Reshaping and transposing into {n_samples_all_channels, num_channels}")
  # subset and reshape
  acc_array = acc_array[:samples_to_subset].reshape(n_samples_all_channels, num_channels)
else:
  console.info(f"Reading {num_channels*timepoints} timepoints. Reshaping to (timepoints, num_channels)")
  acc_array = acc_array.reshape(timepoints, num_channels)


# Filtering
#channel_map = create_channel_map(acc_array, config)
# create filtered and raw array
#bandpass_freqs = config['bandpass']
#
#if has_camera:
#  cam_index = np.where(['cam' in value for value in list(channel_map.values())])
#  cam_array = acc_array[cam_index[0], :]
#  acc_array = np.delete(acc_array, cam_index, axis=0)
#  # save camera
#  suffix = f"_camframes.npy"
#  outfile = os.path.join(ephys_folder, f"{Path(acc_file).stem}{suffix}")
#  console.success(f"Saving camera frames as {outfile}")
#  np.save(outfile, cam_array)
#
#filtered_data = mne.filter.filter_data(acc_array.astype('float64'), 
#                                  f_aq, 
#                                  min(bandpass_freqs), 
#                                  max(bandpass_freqs), 
#                                  verbose=0, n_jobs=2)


#suffix = f"_desc-filt-{min(bandpass_freqs)}-{max(bandpass_freqs)}_eeg.npy"
#outfile = os.path.join(ephys_folder, f"{Path(acc_file).stem}{suffix}")
#console.success(f"Saving eegdata as {outfile}")
#np.save(outfile, filtered_data.astype('float32'))


suffix = f"_acceldata.npy"
outfile = os.path.join(ephys_folder, f"{Path(acc_file).stem}{suffix}")
console.success(f"Saving acceleration data as {outfile}")
np.save(outfile, acc_array.astype('float32'))
