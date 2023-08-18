import numpy as np
import subprocess
import os
import yaml
import sys
from py_console import console
from pathlib import Path
from utils import *
from rlist_files import list_files
# photometry sync
import tdt
import phototdt
from pathlib import Path
import argparse
# filtering
import mne
from mne.io import RawArray
import polars as pl
from scipy.signal import decimate


# Create an argument parser
parser = argparse.ArgumentParser(description='This script reads all _eeg.bin in a folder and binds them together')

# Add arguments for ephys_folder and config_folder
parser.add_argument('--session_folder', help='Path to the session folder (ending in yyyy-mm-dd)')
parser.add_argument('--config_folder', help='Path to the config folder')

# Parse the command-line arguments
args = parser.parse_args()

# Check if ephys_folder argument is provided, otherwise prompt the user
if args.session_folder:
    ephys_folder = args.session_folder
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
console.info(f"Finding configs in {config_folder}")
config = read_config(config_folder)
subject_id = config["subject_id"]
console.info(f"Finding TTLs in {ephys_folder}")
# list the ttl files
ttl_files = list_files(os.path.join(ephys_folder, "ttl"), pattern = ".bin", full_names = True)
num_channels = len(config['ttl_names'])
# ttl data comes demultiplexed in ColumMajor
# dtype for ttl it's np.int8
ttl_array, ttl_samples = read_stack_chunks(ttl_files, num_channels, dtype=np.int8, return_nsamples = True)
# normalize so max values go from 1, 2, 3, 4, 5, 6, 7, 8 to all being 1 
ttl_array = normalize_ttl(ttl_array, method="max")
console.success("TTL data read and normalized")
# TODO: We might want to save the TTL it at some point, but not for now

# We want to chunk the ephys data so that it matches the ttl_data
# replace the ttl_in for eeg
console.info('Finding Matching EEG files')
matching_eeg_files = [val.replace("ttl_in", "eeg") for val in ttl_files]
# replace the /ttl/ for /eeg/
matching_eeg_files = [val.replace('/ttl/', '/eeg/') for val in matching_eeg_files]

# Iterate over the files and combine data
combined_data, eeg_samples = read_stack_chunks(matching_eeg_files, len(config['selected_channels']), dtype = np.float32, return_nsamples = True)
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
    data_down = decimate(combined_data, downsample_factor, ftype='fir')
    channel_map = create_channel_map(data_down, config)
    console.info(f"Provided channel map is {channel_map}")
    eeg_df = pl.DataFrame(data_down.T, schema = channel_map)
    # use the first timestamp of the session for each chunk
    session_id = re.search(r'\d{8}T\d{6}', matching_eeg_files[0]).group()
    outfilename = os.path.join(ephys_folder, f"sub-{subject_id}_ses-{session_id}_alignedeeg.csv.gz")
    eeg_df.write_csv(outfilename)
    console.success(f"Downsampled data written to {outfilename}")


# -------------------------------------------- #
# --------- Now we perform alignment --------- #
# -------------------------------------------- #


def find_mismatch(array1, array2):
    # Check which array is longer
    if len(array1) > len(array2):
        longer_array = array1
        shorter_array = array2
    else:
        longer_array = array2
        shorter_array = array1
        
    # Iterate over the shorter array and compare with the longer array
    for i in range(len(shorter_array)):
        if abs(longer_array[i] - shorter_array[i]) > 1:
            return i
            
    # If all elements in the shorter array match with the corresponding elements in the longer array,
    # check if the last element of the longer array is greater than the last element of the shorter array
    if longer_array[-1] - shorter_array[-1] > 1:
        return len(shorter_array)
    
    # If both arrays match, return -1
    return -1

def fix_tdt_names(string, append): 
  # if the epocs have less than 3 letters, they would be stored as
  # "PC0/" but can only be accessed using "PC0_" on the dict
  # this little helper gives a hand to avoid errors if they happened to occur
  if len(string) < 4:
    return f"{string}{append}"
  else:
    return string

# Finding the photometry file
console.log(f"Choose any Matching Photometry file for session {session_id}")
photometry_file = ui_find_file(title=f"Choose any Matching Photometry file for session {session_id}", initialdir=ephys_folder)
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

#if len(tdt_pulse_onset) != len(pulse_onset):
#  console.warn(f"It seems that the number of pulses on each recording is not equal", severe = True)
#  # This should find the places where 
#  # we do aq_freq + 2 to be safe from off by one errors. We are looking for big jumps
#  jumps = np.where(np.diff(pulse_onset, prepend=0) > config['aq_freq_hz'] + 2)[0]
#  console.info(f'Found jumps in `pulse_onset` at idx: {jumps}. Trying to align with first jump.')
#  # the first jump should be at idx zero
#  alignment_idx = jumps[1]
#  if len(pulse_onset[alignment_idx:]) == len(tdt_pulse_onset):
#    console.info(f'Aligment successful using new idx at {alignment_idx}')
#    eeg_t0_sec = pulse_onset[alignment_idx] / config['aq_freq_hz']
#  else:
#    console.error(f"Could not aling the data. Please check it manually", severe=True)
#    sys.exit(0)
#else:
#  console.success(f"Lengths match on both recordings, aligning to first event")
#  # If everything checks out 
alignment_idx = 0
# in continuous mode, the first TTL file will always have less elements than the first eeg file
# the last TTL file will also have less elements than the last eeg file
# we have to account for that by adding the difference in samples to the pulse_onset
samples_to_add = (np.array(eeg_samples) - np.array(ttl_samples))[0] 
console.info(f"The difference between the first eeg_samples and ttl_samples is {samples_to_add} samples.")
console.info(f"This difference should match the previous shape information. Adding {samples_to_add} to pulse_onset to match eeg_t0_sec")
pulse_onset = pulse_onset + samples_to_add
# now that we accounted for the difference, we can convert to seconds
eeg_t0_sec = pulse_onset[alignment_idx] / config['aq_freq_hz']



# OUTPUTS

params_dict = {
  "eeg_t0_sec" : {"description": "the time in seconds to subtract from the time vector on the ephys recording",
                      "value": float(eeg_t0_sec)
  },
  "photo_max_t" : {"description": "the maximum time in photometry recording in seconds",
                      "value" : max_t

  },
  "alignment_idx" :{"description": "The index on bonsai_pulse_onset used for alignment. pulse_onset[alignment_idx] shoud be eeg_t0_sec. This index will be zero (the first pulse), unless bonsai received more pulses than TDT sent (e.g., TDT recording started and stopped while bonsai kept recording)",
                      "vaue" : int(alignment_idx)

  },
  "tdt_pulses_sent" : {"description": 'the number of pulses sent by TDT',
                          "value": len(tdt_pulse_onset)
  },
  "bonsai_pulses_received" : {"description": "the number of pulses received by bonsai",
                                "value": len(pulse_onset)}
}

# It will have the filename of the first ttl file
params_file = os.path.join(ephys_folder, 'ttl', f"{Path(ttl_files[0]).stem}_alignment_params.yaml")

with open(params_file, 'w') as outfile:
    yaml.safe_dump(params_dict, outfile, default_flow_style=False, sort_keys=False)
console.success(f"Saved alignment params in {params_file}")