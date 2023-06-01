# This contains the old implementation of alignment using batches of sampling data
# we switched into using the first pulse
# we are also checking so that we make sure the number of pulses on each recording is the same

import os
import yaml
import sys
from py_console import console
from pathlib import Path
import numpy as np
import pandas as pd
# photometry sync
import tdt
from utils import *

console.log("Choose TTL file")
root = tk.Tk()
root.withdraw()
ttl_file = ui_find_file(title="Choose TTL file", initialdir=os.path.expanduser("~"), file_type = "npy")
ephys_folder = os.path.dirname(ttl_file)
console.info(f"Working on {ephys_folder}")

config = read_config(ephys_folder)

console.info(f"loading {ttl_file}")
ttl_events = np.load(ttl_file)
ttl_time_file = validate_file_exists(ephys_folder, "ttl_timestamp*")
console.info(f"loading {ttl_time_file}")
ttl_timestamps = pd.read_csv(ttl_time_file)

# verify ttl lenghts
if (ttl_events.shape[1] == ttl_timestamps.shape[0] * config['buffer']['ttl']):
  console.success("ttl_events have correct shape")
else:
  console.warn("ttl_events have incorrect shape", severe=True)


# 1000 Hz is 1024 samples in 1024 milliseconds
# samples are collected using the ttl buffer
# first 60 seconds
n_samples = 60 * config['aq_freq_hz'] 

# find photometry events
photo_ttl_idx = np.where("photometry" in config["ttl_names"])[0]
photo_events = ttl_events[photo_ttl_idx, :].flatten()
photo_first_high = min(np.where(photo_events == 1)[0])
rec_start = photo_events[photo_first_high:(photo_first_high+n_samples)]

#peaks, _ = find_peaks(rec_start, plateau_size=[1, 250])
peaks = np.where(np.diff(rec_start, prepend=0) > 0)[0]
# shift peaks to original idx
peaks = peaks + photo_first_high

multiples_of_batch = peaks/config['buffer']['ttl'] 
distance_from_batch =  multiples_of_batch - np.round(multiples_of_batch)

# send all negative things to infinity and ask for the smallest positive thing
positive_d_from_batch = np.where(distance_from_batch > 0, distance_from_batch, np.inf)
# TODO validate close enough
batch_to_seconds = config['buffer']['ttl']/config['aq_freq_hz']
min_d = positive_d_from_batch.min()
print(f"Distance to batch multiple {min_d}, {min_d * batch_to_seconds} seconds")

if min_d > 0.1:
  console.error(f"Cannot align below threshold of 100 ms", severe=True)
  sys.exit()


# OUTPUTS
# closest_to_batch: the index of the best aligned peak 
# best_aligned_ttl_idx: the index ttl sample
# best_aligned_timestamp_idx: the index of the timestamp to use as new t0

closest_to_batch = int(np.where(distance_from_batch > 0, distance_from_batch, np.inf).argmin())
best_aligned_ttl_idx = int(peaks[closest_to_batch])
best_aligned_timestamp_idx = int(best_aligned_ttl_idx / config['buffer']['ttl'])

params_dict = {
  "closest_to_batch" : {"description": "the index of the best aligned peak",
                      "value": closest_to_batch
  },
  "best_aligned_ttl_idx" : {"description": 'the index of the best aligned ttl sample',
                          "value": best_aligned_ttl_idx
  },
  "best_aligned_timestamp_idx" : {"description": "the index of the ttl timestamp to use as new t0",
                                "value": best_aligned_timestamp_idx}
}

params_file = os.path.join(ephys_folder, f"{Path(ttl_file).stem}_alignment_params.yaml")

with open(params_file, 'w') as outfile:
    yaml.safe_dump(params_dict, outfile, default_flow_style=False, sort_keys=False)
console.success(f"Saved alignment params in {params_file}")