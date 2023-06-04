import yasa
from staging import SleepStaging
import mne
from mne.io import RawArray
from staging import SleepStaging
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl
from rlist_files import list_files
from py_console import console
from utils import *


def predict_electrode(eeg, emg, epoch_sec = 2):
  info =  mne.create_info(["eeg","emg"], 
                          sf, 
                          ch_types='misc', 
                          verbose=False)
  raw_array = RawArray(np.vstack((eeg, 
                                  emg)),
                       info, verbose=False)
  sls = SleepStaging(raw_array,
                     eeg_name="eeg", 
                     emg_name="emg")
  # this will use the new fit function
  sls.fit(epoch_sec=epoch_sec)
  # the auto will use these features
  # "/home/matias/anaconda3/lib/python3.7/site-packages/yasa/classifiers/clf_eeg+emg_lgb_0.5.0.joblib"
  predicted_labels = sls.predict(path_to_model="clf_eeg+emg_lgb_gbdt_custom.joblib")
  proba = sls.predict_proba()
  return predicted_labels, proba

def plot_spectrogram(eeg, hypno, epoch_sec = 2):
  # upsample to data
  label_df = yasa.hypno_upsample_to_data(hypno,
  sf_hypno=1/epoch_sec, 
  data=eeg, sf_data=sf)
  fig = yasa.plot_spectrogram(eeg,
                      hypno=label_df, 
                      win_sec = 10,
                      sf=sf,
                      # default is 'RdBu_r'
                      # cmap='Spectral_r',
                      # manage the scale contrast, larger values better contrast
                      trimperc = 1)
  fig.show()


console.log("Choose downsampled EEG file")
eeg_file = ui_find_file(title="Choose downsampled EEG file", initialdir=os.path.expanduser("~"))
# eeg_file = list_files(pattern  ="eegdata_desc-down-*_eeg.csv.gz")
eeg_df = pl.read_csv(eeg_file)
sf = 100
results = {}

for column in eeg_df.columns:
    if column.startswith('EEG'):
        eeg = eeg_df[column].to_numpy()
        
        # Calculate EMG difference for each EEG channel
        emg_diff = eeg_df['EMG2'] - eeg_df['EMG1']
        
        # Perform prediction and plot spectrogram for the EEG channel
        hypno, proba = predict_electrode(eeg, emg_diff)
        plot_spectrogram(eeg, hypno)
        
        # Store hypno and proba in the results dictionary under the column key
        results[column] = {"hypno": hypno, "proba": proba}