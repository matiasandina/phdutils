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
from bonsai_dat_to_npy_eeg import *

def align_ttl_to_eeg(eeg_samples, ttl_samples):
    '''
    This function will check what scenario for ttls we have
    It assumes you are passing continuous eeg_samples and ttl_samples as identified previously
    Possible Scenarios:
        X: we can have co-start ends before,  starts after ends before, starts after co-terminate.
        XY: Here the scenario is that the ttl recording starts after and co-terminates in X because of continuity, and ends before or co-terminates on Y.
        X[Full]nY: This is an extension of the XY scenario with n Full chunks where eeg_samples[n] == ttl_samples[n] because the recording is continuous.
    '''

    # Check that you provided samples
    if len(eeg_samples) == 0 or len(ttl_samples) == 0:
        raise ValueError("Error: No samples provided")
    # Check that they are matching in length
    assert len(eeg_samples) == len(ttl_samples), "Samples must be the same length"

    def check_co_terminate(chunk_types):
        # co-terminate examples:
            # Here both chunks co-terminate
            # eeg_samples = [3600128, 3600128]
            # ttl_samples = [399360, 3600128]
            # chunk_types = ['Partial', 'Full']
            # Here first chunk co-terminates (assumption of continuity) second chunk does not
            # eeg_samples = [3600128, 3600128]
            # ttl_samples = [399360, 1234]
            # chunk_types = ['Partial', 'Partial']
        co_terminate = []
        for idx, chunk in enumerate(chunk_types):
            if idx == 0: 
                if len(chunk_types) > 1:
                    # we assume co-terminate due to continuity
                    co_terminate.append(True)
                else:
                    # This case is actually not possible to be determined
                    # we shouldn't pass len(chunk_types) > 1 ?
                    # However, it's not possible to stop ttl_rec
                    # at the same time that files are auto chunked by bonsai
                    # period of 'HH:MM:SS'. 
                    # Intuition tells me the only case this would happen is if bonsai crashes in the first chunk
                    # returning False will only be a problem in that case
                    co_terminate.append(False)
            else:
                # Here the only way a chunk can co-terminate
                # is if we found it to have the same number of samples
                co_terminate.append(chunk == "Full")
        return co_terminate

    def check_co_start(chunk_types):
        # co-start examples:
            # Here the first chunk does not co-start, second one does
            # eeg_samples = [3600128, 3600128]
            # ttl_samples = [399360, 3600128]
            # chunk_types = ['Partial', 'Full']
            # Here you have the same situation becasue assumption of continuity
            # eeg_samples = [3600128, 3600128]
            # ttl_samples = [399360, 1234]
            # chunk_types = ['Partial', 'Partial']
        co_start = []
        for idx, chunk in enumerate(chunk_types):
            if idx == 0:
                # first chunk will only co-start if chunk_type is "Full"
                co_start.append(chunk == 'Full')
            else:
                # We assume continuity if more than one chunk
                co_start.append(True)
        return co_start

    # Check what type of matching we have
    # Matching will be full if eeg and ttl 
    # have the same number of samples for a chunk 
    # Some examples:
        # ['Partial', 'Full', 'Full', 'Partial']
        # or ['Partial', 'Full', 'Full', 'Full']
        # or ['Partial', 'Partial']
        # or ['Partial] <- we have to handle len() == 1 case separately
        # It's not possible to align without external information of when events happened
    chunk_types = ['Full' if eeg == ttl else 'Partial' for eeg, ttl in zip(eeg_samples, ttl_samples)]
    co_terminate = check_co_terminate(chunk_types)
    co_start = check_co_start(chunk_types)
    sample_differences = [eeg - ttl for eeg, ttl in zip(eeg_samples, ttl_samples)]
    
    match len(eeg_samples):
        case 1:
            output = {
                "continuous_chunks": 1,
                "samples_before_ttl": [],
                "samples_after_ttl": [],
            }
        case 2:
            output = {
                "continuous_chunks": 2,
                "samples_before_ttl": sample_differences[0],
                "samples_after_ttl": sample_differences[-1],
            }
        case _:
            output = {
                "continuous_chunks": len(eeg_samples),
                "samples_before_ttl": sample_differences[0],
                "samples_after_ttl": sample_differences[-1]
            }
    
    output["chunk_types"] = chunk_types
    output["co_start"] = co_start
    output["co_terminate"] = co_terminate
    return output

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


def create_params_dict(eeg_t0_sec, max_t, alignment_idx, tdt_pulse_onset, pulse_onset):
    return {
        "eeg_t0_sec": {
            "description": "the time in seconds to subtract from the time vector on the ephys recording",
            "value": float(eeg_t0_sec)
        },
        "photo_max_t": {
            "description": "the maximum time in photometry recording in seconds",
            "value": max_t
        },
        "alignment_idx": {
            "description": "The index on bonsai_pulse_onset used for alignment.",
            "value": int(alignment_idx)
        },
        "tdt_pulses_sent": {
            "description": 'the number of pulses sent by TDT',
            "value": len(tdt_pulse_onset)
        },
        "bonsai_pulses_received": {
            "description": "the number of pulses received by bonsai",
            "value": len(pulse_onset)
        }
    }

def save_alignment_params(params, output_folder):
  params_file = os.path.join(
     output_folder, "alignment_params.yaml")
  with open(params_file, 'w') as outfile:
     yaml.safe_dump(
        params, 
        outfile, 
        default_flow_style=False, 
        sort_keys=False)
  console.success(f"Saved alignment params in {params_file}")

def align_single_chunk(ttl_chunk, config, output_folder):
  '''
  This function should align a single chunk by matching with respective eeg_files
  '''
  session_id = parse_bids_session(ttl_chunk[0])
  console.info(f"Processing aligned recording with session_id: {session_id}")
  # ttl data comes demultiplexed in ColumMajor
  # dtype for ttl it's np.int8
  num_channels = len(config['ttl_names'])
  ttl_array, ttl_samples = read_stack_chunks(ttl_chunk, 
                                             num_channels,
                                             dtype=np.int8, 
                                             return_nsamples = True)
  # normalize so max values go from 1, 2, 3, 4, 5, 6, 7, 8 to all being 1 
  ttl_array = normalize_ttl(ttl_array, method="max")
  console.success("TTL data read and normalized")
  # TODO: We might want to save the TTL it at some point, but not for now

  # We want to chunk the ephys data so that it matches the ttl_data
  # replace the ttl_in for eeg
  console.info('Finding Matching EEG files')
  matching_eeg_files = [val.replace("ttl_in", "eeg") for val in ttl_chunk]
  # replace the /ttl/ for /eeg/
  matching_eeg_files = [val.replace('/ttl/', '/eeg/') for val in matching_eeg_files]
  # Iterate over the files and combine data
  #combined_data, eeg_samples = read_stack_chunks(matching_eeg_files, len(config['selected_channels']), dtype = np.float32, return_nsamples = True)
  # filter, downsample, and save
  eeg_downsampled, nsamples_dict = filter_down_bonsai_eeg(config=config, file_list=matching_eeg_files, output_folder=output_folder)
  
  # -------------------------------------------- #
  # --------- Now we perform alignment --------- #
  # -------------------------------------------- #
  # Finding the photometry file
  # This must be done manually because we have to find the proper TDT file
  console.log(f"Choose any Matching Photometry file for session {session_id}")
  photometry_file = ui_find_file(title=f"Choose any Matching Photometry file for session {session_id}", initialdir=ephys_folder)
  tdt_folder = os.path.dirname(photometry_file)
  console.info(f"Selected TDT folder is {tdt_folder}")
  
  # we read a very very small portion only to get the info
  block = tdt.read_block(tdt_folder, t1=0, t2=1)
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
  session_eeg_file = list_files(output_folder, pattern=session_id, full_names = True)[0]
  eeg_samples = nsamples_dict[session_eeg_file]
  alignment_dir  = align_ttl_to_eeg(eeg_samples, ttl_samples)
  if alignment_dir['continuous_chunks'] == 1:
    console.warn('Cannot align chunks of length 1. No way to assume co-startt/co-termination')
    console.error('Exiting', severe=True)
    sys.exit(0)
  samples_to_add = alignment_dir['samples_before_ttl']
  console.info(f"The difference between the first eeg_samples and ttl_samples is {samples_to_add} samples.")
  console.info(f"This difference should match the previous shape information. Adding {samples_to_add} to pulse_onset to match eeg_t0_sec")
  pulse_onset = pulse_onset + samples_to_add
  # now that we accounted for the difference, we can convert to seconds
  eeg_t0_sec = pulse_onset[alignment_idx] / config['aq_freq_hz']
  
  # OUTPUTS
  # Create params dict for the current session
  current_params = create_params_dict(eeg_t0_sec, max_t, alignment_idx, tdt_pulse_onset, pulse_onset)
  console.success(f"Finished Alignment for session {session_id}")
  return session_eeg_file, current_params


def run_alignment(ephys_folder, config_folder):
  # Now you can use ephys_folder and config_folder in your script
  console.info(f"Finding configs in {config_folder}")
  config = read_config(config_folder)
  # Perform some checks in the config
  assert config['down_freq_hz'] is not None, "No down_freq_hz in config. Exiting function"
  assert config["aq_freq_hz"] > config["down_freq_hz"], f"{config['aq_freq_hz']} must be greater than {config['down_freq_hz']}"

  subject_id = config["subject_id"]
  console.info(f"Finding TTLs in {ephys_folder}")
  # list the ttl files
  ttl_files = list_files(os.path.join(ephys_folder, "ttl"), pattern = ".bin", full_names = True)
  bonsai_timer_period = datetime.datetime.strptime(config["bonsai_timer_period"], "%H:%M:%S")
  expected_delta_sec = datetime.timedelta(hours = bonsai_timer_period.hour, minutes= bonsai_timer_period.minute, seconds = bonsai_timer_period.second).total_seconds()
  expected_delta_min = expected_delta_sec / 60

  # there shouldn't be discontinuities, but shit happens sometimes and there are
  ttl_chunks = chunk_file_list(ttl_files, expected_delta_min, 1)
  if len(ttl_chunks) > 1:
    console.warn(" #### Found discontinuity in TTL, RECORDING WAS NOT CONTINUOUS ####  ")

  output_folder = os.path.join(ephys_folder, "aligned", "eeg")
  if not os.path.isdir(output_folder):
    console.log(f"Creating Directory: {output_folder}")
    os.makedirs(output_folder)
  # create output dir
  all_sessions_params = {}
  # Loop on each chunk 
  for chunk_idx, ttl_chunk in enumerate(ttl_chunks):
    # Align chunk and return params
    session_eeg_file, current_params = align_single_chunk(
       ttl_chunk=ttl_chunk, 
       config=config,
       output_folder=output_folder)
    # Store align params in output dictionary
    all_sessions_params[session_eeg_file] = current_params
    
  # save params dict
  save_alignment_params(all_sessions_params, os.path.dirname(output_folder))
  return all_sessions_params

if __name__ == "__main__":
  # Create an argument parser
  parser = argparse.ArgumentParser(description='This script performs alignment betweeen EEG data and TTL pulses sent via TDT')
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
  
  run_alignment(ephys_folder=ephys_folder, config_folder=config_folder)