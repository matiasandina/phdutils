import os
import numpy as np
import scipy.io
from py_console import console
import mne
from utils import *

# TODO: Accusleep wants double precision so we have to use
# .astype(np.float64)
# At some point we should get rid of this if no longer using Accusleep

# Load data to work with
console.info("Choose filtered EEG file")
root = tk.Tk()
root.withdraw()
eeg_file = ui_find_file(title="Choose filtered eeg file", initialdir=os.path.expanduser("~"), file_type = "npy")

root_dir = os.path.dirname(eeg_file)
folder_name = os.path.join(root_dir, "eeg_mat")
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

config = read_config(root_dir)

# find the emg channels in the config
emg_channels = find_channels(config, "EMG")

if emg_channels is None:
  console.error("Indices for EMG Channels not found in channel map before filtering.\nCheck your data!\nExiting program.", severe=True)
else:
  console.success(f"Found EMG channels at idx {emg_channels}.")

# load eeg and save it
eeg = np.load(eeg_file)
for channel in range(eeg.shape[0]):
    fn = os.path.join(folder_name, f"eeg_channel{channel:02d}.mat")
    if channel in emg_channels:
        EMG = eeg[channel, :].astype(np.float64)
        scipy.io.savemat(fn, {"EMG": EMG})
        console.success(f"saved {fn}")
    else:
        EEG = eeg[channel, :].astype(np.float64)
        scipy.io.savemat(fn, {"EEG": EEG})
        console.success(f"saved {fn}")

# create difference emg
emg_diff = np.subtract(eeg[emg_channels[0]], eeg[emg_channels[1]])

fn = os.path.join(folder_name, "emg_diff.mat")
scipy.io.savemat(fn, {"EMG": emg_diff.astype(np.float64)})
console.success(f"saved {fn}")