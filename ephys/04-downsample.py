import numpy as np
import subprocess
import os
import yaml
import sys
from py_console import console
from pathlib import Path
from utils import *
import pandas as pd
# downsampling
from scipy.signal import decimate

console.log("Choose EEG file")
eeg_file = ui_find_file(title="Choose filtered file", initialdir=os.path.expanduser("~"), file_type = "npy")
ephys_folder = os.path.dirname(eeg_file)
console.info(f"Working on {ephys_folder}")

config_file = validate_file_exists(ephys_folder, "config.yaml")
config = read_yaml(config_file)

eeg_array = np.load(eeg_file)

#try:
#	params_file = validate_file_exists(ephys_folder, "alignment_params.yaml")
#	params = read_yaml(params_file)
#	best_aligned_ttl_idx = params['best_aligned_ttl_idx']['value']
#	eeg_array = eeg_array[:, best_aligned_ttl_idx:]
#	console.success("Alignment found, subsetting before downsampling")
#except:
#	console.warn("No alignment found, using all data")

# 'fir' is super important, all else too slow and breaking
if config['down_freq_hz'] is None:
	console.info("No down_freq_hz in config. Exit without downsampling")
	sys.exit()
else:
	assert config["aq_freq_hz"] > config["down_freq_hz"], f"{config['aq_freq_hz']} must be greater than {config['down_freq_hz']}"
	downsample_factor = int(config["aq_freq_hz"]/config["down_freq_hz"])
	console.info(f"Downsampling with factor {downsample_factor}")
	data_down = decimate(eeg_array, downsample_factor, ftype='fir')
	eeg_df = pd.DataFrame(data_down.T)
	channel_map = create_channel_map(data_down, config)
	console.info(f"Provided channel map is {channel_map}")
	eeg_df = eeg_df.set_axis(channel_map, axis=1, copy=False)
	outfilename = os.path.join(ephys_folder, f"_desc-down-{downsample_factor}_eeg.csv.gz")
	eeg_df.to_csv(outfilename, index=False, header=True)
	console.success(f"Downsampled data written to {outfilename}")
