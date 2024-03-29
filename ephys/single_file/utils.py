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

def read_yaml(filename):
  with open(filename, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
  return cfg

# check for config available
def read_config(ephys_folder):
  config_file = list_files(path = ephys_folder, pattern="config.yaml", full_names=True)
  if not config_file:
    console.error(f"config.yaml not found in {ephys_folder}", severe=True)
    sys.exit()
  elif len(config_file) > 1:
    console.error(f"{ephys_folder} contains more than on config.yaml: {config_file}", severe=True)
    sys.exit()
    # unlist using first element, assume only one match...
  else:
    config_file = config_file[0]
    config = read_yaml(config_file)
  return config

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

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
