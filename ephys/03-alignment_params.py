import os
import yaml
import sys
from py_console import console
from pathlib import Path
import numpy as np
import pandas as pd
# photometry sync
import tdt
import phototdt
from utils import *

 
def fix_tdt_names(string, append): 
  # if the epocs have less than 3 letters, they would be stored as
  # "PC0/" but can only be accessed using "PC0_" on the dict
  # this little helper gives a hand to avoid errors if they happened to occur
  if len(string) < 4:
    return f"{string}{append}"


console.log("Choose TTL file")
ttl_file = ui_find_file(title="Choose TTL file", initialdir=os.path.expanduser("~"), file_type = "npy")
ephys_folder = os.path.dirname(ttl_file)
console.info(f"Working on {ephys_folder}")
# find config
config = read_config(ephys_folder)
# Ask for TDT photometry data
console.log("Choose any Photometry file")
photometry_file = ui_find_file(title="Choose any Photometry file", initialdir=ephys_folder)
tdt_folder = os.path.dirname(photometry_file)
console.info(f"Selected TDT folder is {tdt_folder}")

# we read a very very small portion only to get the info
block = tdt.read_block(tdt_folder, t1=0, t2=0.5)
# get the duration of the photometry recording
max_t = phototdt.get_total_duration(block)
# we also read the events, these might lead to errors when changing the name of the store
pulse_sync_name = config['pulse_sync']
tdt_pulse_onset = tdt.read_block(tdt_folder, store = fix_tdt_names(pulse_sync_name, '/')).epocs[fix_tdt_names(pulse_sync_name, '_')].onset
# We cannot use the last offset here because it's inf, but the difference is < pulse's width here
tdt_epoc_duration = tdt_pulse_onset[-1] - tdt_pulse_onset[0]

console.info(f"loading {ttl_file}")
ttl_array = np.load(ttl_file)
photo_ttl_idx = np.where("photometry" in config["ttl_names"])[0]
photo_events = ttl_array[photo_ttl_idx, :].flatten()
pulse_onset = np.where(np.diff(photo_events, prepend=0) > 0)[0]
pulse_offset = np.where(np.diff(photo_events, prepend=0) < 0)[0]
# These things should give the same duration
tdt_recording_duration_sec = (pulse_offset[-1] - pulse_onset[0]) / 1000
# this difference should be close to zero!
tdt_bonsai_diff = tdt_recording_duration_sec - block.info.duration.total_seconds()

# Print some info
console.info(f"Duration difference: block info - bonsai pulses = {tdt_bonsai_diff} seconds")
console.info(f"Duration difference: block info - tdt sent epocs = {block.info.duration.total_seconds() - tdt_epoc_duration} seconds")
# There's something weird with a difference of pulses between counting methods
console.info(f"TDT sent: {len(tdt_pulse_onset)} pulses")
console.info(f"Bonsai received: {len(pulse_onset)} pulses")

if len(tdt_pulse_onset) != len(pulse_onset):
  console.warn(f"It seems that the number of pulses on each recording is not equal", severe = True)
  # This should find the places where 
  # we do aq_freq + 2 to be safe from off by one errors. We are looking for big jumps
  jumps = np.where(np.diff(pulse_onset, prepend=0) > config['aq_freq_hz'] + 2)[0]
  # the first jump should be at idx zero
  alignment_idx = jumps[1]
  console.info(f'Found jumps in `pulse_onset` at idx: {jumps}. Trying to align with first jump.')
  if len(pulse_onset[alignment_idx:]) == len(tdt_pulse_onset):
    console.info(f'Aligment successful using new idx at {alignment_idx}')
    eeg_t0_sec = pulse_onset[alignment_idx] / config['aq_freq_hz']
  else:
    console.error(f"Could not aling the data. Please check it manually", severe=True)
    sys.exit(0)
else:
  console.success(f"Lengths match on both recordings, aligning to first event")
  # If everything checks out 
  alignment_idx = 0
  eeg_t0_sec = pulse_onset[alignment_idx] / config['aq_freq_hz']



# OUTPUTS

params_dict = {
  "eeg_t0_sec" : {"description": "the time in seconds to subtract from the time vector on the ephys recording",
                      "value": float(eeg_t0_sec)
  },
  "photo_max_t" : {"description": "the maximum time in photometry recording in seconds",
                      "value" : max_t

  },
  "alignment_idx" :{"description": "The index on bonsai_pulse_onset used for alignment. pulse_onset[alignment_idx] shoud be eeg_t0_sec. This will be zero, unless bonsai received more pulses than TDT sent (e.g., TDT recording started and stopped while bonsai kept recording)",
                      "vaue" : int(alignment_idx)

  },
  "tdt_pulses_sent" : {"description": 'the number of pulses sent by TDT',
                          "value": len(tdt_pulse_onset)
  },
  "bonsai_pulses_received" : {"description": "the number of pulses received by bonsai",
                                "value": len(pulse_onset)}
}

params_file = os.path.join(ephys_folder, f"{Path(ttl_file).stem}_alignment_params.yaml")

with open(params_file, 'w') as outfile:
    yaml.safe_dump(params_dict, outfile, default_flow_style=False, sort_keys=False)
console.success(f"Saved alignment params in {params_file}")