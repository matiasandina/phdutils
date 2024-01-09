import yaml
import os
import sys
from py_console import console
import numpy as np
import pandas as pd
import glob
import datetime
from rlist_files.list_files import list_files
import subprocess
import re
import mne
import pytz
import pathlib

def get_last_modif_utc(file_path):
    fname = pathlib.Path(file_path)
    assert fname.exists(), f'No such file: {fname}' 
    # get the modification time
    # it's stored under st_mtime on the stat() return object
    # mind datetime.timezone.utc. 
    # We define tz output from st_mtime to avoid running into issues with locale
    mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime, tz=datetime.timezone.utc)
    return mtime

def get_last_modif(file_path, tz=None):
    '''
    This is a glorified wrapper to convert the output of get_last_modif_utc() to a specific timezone
    It's unlikely that you might be running this code on a separate tz from creation (EST)
    and since files are timestamped in the creation zone (EST) so chaning tz here might create issues
    with math.
    Only use tz if you are aware of these issues
    '''
    if tz is None:
        tz = pytz.timezone('America/New_York')        
    else:
        tz = pytz.timezone(tz)
    utc_mtime = get_last_modif_utc(file_path)
    return utc_mtime.astimezone(tz=tz)


def parse_bids_subject(string: str):
  return os.path.basename(string).split("_")[0].replace("sub-", "")

def parse_bids_session(string: str):
  return os.path.basename(string).split("_")[1].replace("ses-", "")

def bids_naming(session_folder, subject_id, session_date, filename):
  session_date = session_date.replace("-", "")
  return os.path.join(session_folder, f"sub-{subject_id}_ses-{session_date}_{filename}")


def read_yaml(filename):
  with open(filename, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
  return cfg

# check for config available
def read_config(config_folder):
  config_file = list_files(path = config_folder, pattern="config.yaml", full_names=True)
  if not config_file:
    console.error(f"config.yaml not found in {config_folder}", severe=True)
    sys.exit()
  elif len(config_file) > 1:
    console.error(f"{config_folder} contains more than on config.yaml: {config_file}", severe=True)
    sys.exit()
    # unlist using first element, assume only one match...
  else:
    config_file = config_file[0]
    config = read_yaml(config_file)
  return config

# Function to check if a string can be parsed as a date
def is_valid_date(date_string):
    try:
        datetime.datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

def chunk_file_list(file_list, expected_delta_min, discontinuity_tolerance):
    """
    Chunk the given file list based on discontinuity in timestamps.

    Parameters:
        file_list (list): List of file names.
        expected_delta_min (int): Expected time difference between consecutive files in minutes.
        discontinuity_tolerance (int): Tolerance for discontinuity in minutes.

    Returns:
        list: List of chunks, where each chunk contains continuous files based on timestamps.

    """
    if not file_list:
      return []
    # Extract timestamps from file names
    timestamps = [re.search(r'\d{8}T\d{6}', file).group() for file in file_list]
    # Convert timestamps to datetime objects
    times = [datetime.datetime.strptime(timestamp, '%Y%m%dT%H%M%S') for timestamp in timestamps]
    # Convert datetime objects to numeric timestamps
    timestamps_numeric = [time.timestamp() for time in times]
    # Calculate time differences between consecutive timestamps
    time_diffs = np.diff(timestamps_numeric) / 60
    # Identify discontinuities based on the expected delta and tolerance
    discontinuity_tolerance_lower = expected_delta_min - discontinuity_tolerance
    discontinuity_tolerance_upper = expected_delta_min + discontinuity_tolerance
    # Find indices where time differences are outside the tolerance range
    discontinuous_indices = np.where(
        (time_diffs < discontinuity_tolerance_lower) | 
        (time_diffs > discontinuity_tolerance_upper)
    )[0]

    # Adjust indices to correctly split the array
    discontinuous_indices += 1

    # Split the file list into chunks based on the discontinuous indices
    chunks = [file_list[i:j] for i, j in zip([0] + list(discontinuous_indices), list(discontinuous_indices) + [len(file_list)])]
    return chunks


def read_stack_chunks(file_chunk, num_channels, dtype=np.float32, return_nsamples = False):
  num_files = len(file_chunk)
  combined_data = [None] * num_files
  # we need to store the number of nsamples for alignment purposes
  nsamples = []
  for file_idx, file in enumerate(file_chunk):
    
    console.log(f"Read data from file {file} ({file_idx+1}/{num_files})")
    # data is stored as np.float32
    eeg_array = np.fromfile(file, dtype=dtype)
    # first integer division, then blowup.
    n_samples_all_channels = eeg_array.shape[0] // num_channels
    nsamples.append(n_samples_all_channels)
    console.info(f"Reading all dataset. Reshaping and transposing into {num_channels, n_samples_all_channels}")
    # subset and reshape
    eeg_array = eeg_array.reshape(n_samples_all_channels, num_channels).T

    # remove camera stamp if present
    #if has_camera:
    #  cam_index = np.where(['cam' in value for value in list(channel_map.values())])
    #  cam_array = eeg_array[cam_index[0], :]
    #  eeg_array = np.delete(eeg_array, cam_index, axis=0)
    #  # save camera
    #  suffix = f"_camframes.npy"
    #  outfile = os.path.join(ephys_folder, f"{Path(eeg_file).stem}{suffix}")
    #  console.success(f"Saving camera frames as {outfile}")
    #  np.save(outfile, cam_array)

    #if file_idx == 0:
        # For the first file, initialize combined_data with the shape of eeg_array
        # concat enforces same dimensions in all arrays
        #combined_data = np.expand_dims(eeg_array, axis=0)
    #else: 
        # For subsequent files, expand dimensions of eeg_array before concatenating with combined_data
        #combined_data = np.concatenate((combined_data, np.expand_dims(eeg_array, axis=0)), axis=0)
      # combine data into the proper place
    combined_data[file_idx] = eeg_array
  #console.log(f"Combined Data has the shape (files, channels, timepoints): {combined_data.shape}")

  # Stack files belonging to a chunk horizontally
  combined_data = np.hstack(combined_data)
  console.info(f"Stacked data horizontally into {combined_data.shape}")
  if return_nsamples:
    return combined_data, nsamples
  else:
    return combined_data

def filter_data(data, config, save=False, outpath = None):
  f_aq = config["aq_freq_hz"]
  # Filtering
  channel_map = create_channel_map(data, config)
  # create filtered and raw array
  bandpass_freqs = config['bandpass']['eeg']
  emg_bandpass = config['bandpass']['emg']
  # find the emg channels
  emg_channels = find_channels(config, "EMG")

  if emg_channels is None:
    console.error("Indices for EMG Channels not found in channel map before filtering.\nCheck your data!\nExiting program.", severe=True)
    sys.exit(0)
  else:
    console.success(f"Found EMG channels at idx {emg_channels}. Bandpassing with {emg_bandpass}")

  # we will not filter emg_channels
  eeg_filter_idx = [i for i in range(data.shape[0]) if i not in emg_channels]

  # filter eegs
  filtered_data = mne.filter.filter_data(data.astype('float64'), 
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

  if save:
    assert outpath is not None, "outpath is mising, cannot save"
    # Save output from filtering_script
    #suffix = f"_desc-filt-{min(bandpass_freqs)}-{max(bandpass_freqs)}_eeg.npy"
    #outfile = os.path.join(ephys_folder, f"{Path(eeg_file).stem}{suffix}")
    console.success(f"Saving eegdata as {outpath}")
    np.save(outpath, filtered_data.astype('float32'))

  return filtered_data


def find_channels(config, pattern):
  # np.char.find will return -1 if pattern not found#
  # we ask for anything other than that to make it boolean
  matches = np.where(np.char.find(config['channel_names'], pattern) != -1)[0]
  if len(matches) == 0:
    console.warn(f"Pattern `{pattern}` not found in channel names {config['channel_names']}")
    return None
  else:
    return matches 

def create_channel_map(eeg_array, config, end = "extend"):
  channel_map = {}
  n_channels = eeg_array.shape[0]
  n_named_chan = len(config['selected_channels'])
  if  n_named_chan < n_channels:
    print(f"Data has {n_channels} channels but only {n_named_chan} named channels")
    # TODO: add "prepend option"
    if end == "extend":
      print("Adding channel(s) for camera(s) at the end")
      cam_names = [f"cam{i}" for i in range(1, n_channels - n_named_chan + 1)]
      new_channel_names = list(config['channel_names'])
      new_channel_names.extend(cam_names)
      for key, value in enumerate(new_channel_names):
        channel_map[key] = value 
    return(channel_map)
  else:
    return(config['channel_names'])

def validate_file_exists(folder, pattern):
  files_match = glob.glob(os.path.join(folder, pattern))
  if not files_match:
    console.error(f"{pattern} not found in {folder}", severe=True)
    sys.exit()
  else:
    if len(files_match) > 1:
      console.warning(f"{pattern} in more than a signle file")
      print(files_match)
      console.error(f"Stopping for pattern fixing")
      sys.exit()
    # unlist using first element, assume only one match...
    filepath = files_match[0]
    return filepath

def normalize_ttl(ttl_matrix, method="max"):
  if method == "max":
    max_per_channel = ttl_matrix.max(axis=1, keepdims=True)
    # remove the zeros so we can devide
    max_per_channel = np.where(max_per_channel == 0, 1, max_per_channel)
    out = ttl_matrix / max_per_channel
    return(out)

def find_pulse_onset(ttl_file, ttl_idx, timestamps_file, buffer, round=False):
  """
  This function reads the ttl pulse file
  Subsets the ttl_file array on ttl_idx
  buffer is sf / 4
  Finds pulse onset by calling np diff and looking for the moments where np.diff is positive
  There's two ways to call this. You can either return the rounded down timestamp (round=True) 
  # or interpolate from the closest timestamp assuming constant sampling rate.
  Rerturns the timestamps according to sampling frequency (sf)
  """
  sf = 4 * buffer
  ttl_events = np.load(ttl_file)
  # todo find in config
  photo_events = ttl_events[ttl_idx, :].flatten()
  pulse_onset = np.where(np.diff(photo_events, prepend=0) > 0)[0]
  # get division and remainder
  div_array = np.array([divmod(i, buffer) for i in pulse_onset])
  # TODO div_array[:, 0] has the rounded version
  # timestamps.iloc[div_array[:, 0], :] + sampling_period * div_array[:, 1] is the way to calculate the proper timestamp
  # this assumes constant sampling rate between known timestamps
  timestamps = pd.read_csv(timestamps_file)
  # TODO: not sure this works for all sf 
  sampling_period_ms = 1/sf * 1000
  out = timestamps.iloc[div_array[:,0], :].copy()
  if round:
    return out
  else:
    out.iloc[:, 0] = out.iloc[:, 0] + sampling_period_ms * div_array[:, 1]
    dt = (sampling_period_ms * div_array[:, 1]).astype('timedelta64[ms]')
    out.iloc[:, 1] = pd.to_datetime(out.iloc[:,1]) + dt
    return out


def ui_find_file(title=None, initialdir=None, file_type=None):
    """
    Find a file using a GUI.
    :param title: Title of the dialog.
    :param initialdir: Initial directory.
    :param file_type: File type.
    :return: File path.
    """
    import os
    from PyQt6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        is_new_instance = True
    else:
        is_new_instance = False

    if title is None:
        title = 'Find a file'
    if initialdir is None:
        initialdir = os.getcwd()
    if file_type is None:
        file_filter = 'All files (*.*)'
    else:
        file_filter = f"{file_type} files (*.{file_type})"
        #file_filter = [f"{file} files (*.{file})" for file in file_type]
        #file_filter = ';;'.join(file_filter)

    file_path, _ = QFileDialog.getOpenFileName(None, title, initialdir, file_filter)

    if file_path:
        # no multiple files for now...file_path will be a string
        #file_path = file_path[0]
        print(f"Selected {file_path}")
        result = file_path
    else:
        print("No file selected")
        result = None

    if is_new_instance:
        app.quit()

    return result
