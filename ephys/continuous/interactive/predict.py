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
import argparse

def get_max_probabilities(results):
    electrode_probabilities = {}
    for electrode, data in results.items():
        proba = data['proba'].max(axis=1)
        electrode_probabilities[electrode] = proba
    return pd.DataFrame(electrode_probabilities)


def aggregate_hypno_predictions(results):
    electrode_hypno = {}
    for electrode, data in results.items():
        hypno = data['hypno']
        electrode_hypno[electrode] = hypno
    return pd.DataFrame(electrode_hypno)


def get_most_frequent_value(predictions_df):
    from scipy.stats import mode
    # Calculate the most frequent value (mode) across all electrodes for each epoch
    most_frequent, _ = mode(predictions_df.values, axis=1, keepdims = False)
    return most_frequent

def consensus_prediction(predictions_df, max_probabilities_df):
    # Ensure both DataFrames have the same indexes
    assert np.all(predictions_df.index == max_probabilities_df.index), "Indexes must match."
    # Get unique categories from the predictions DataFrame
    categories = np.unique(predictions_df.values.ravel())
    # Initialize DataFrame to store weighted votes for each category
    weighted_votes_df = pd.DataFrame(index=predictions_df.index)
    # Calculate weighted votes for each category
    for category in categories:
        # Create DataFrame with binary values indicating whether each element matches the current category
        binary_df = (predictions_df == category).astype(int)
        # Calculate weighted votes by multiplying binary DataFrame element-wise with max_probabilities_df
        weighted_votes_df[category] = (binary_df * max_probabilities_df).sum(axis=1)
    # Consensus predictions are the categories with maximum weighted vote
    consensus_predictions = weighted_votes_df.idxmax(axis=1)
    return consensus_predictions


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

def  predict_and_save(epoch_sec):
    console.log("Choose downsampled EEG file")
    eeg_file = ui_find_file(title="Choose downsampled EEG file", initialdir=os.path.expanduser("~"))
    session_folder = os.path.dirname(eeg_file)
    animal_id = parse_bids_subject(os.path.basename(eeg_file))
    session = parse_bids_session(os.path.basename(eeg_file))
    # eeg_file = list_files(pattern  ="eegdata_desc-down-*_eeg.csv.gz")
    eeg_df = pl.read_csv(eeg_file)
    results = {}

    for column in eeg_df.columns:
        if column.startswith('EEG'):
            eeg = eeg_df[column].to_numpy()
            
            # Calculate EMG difference for each EEG channel
            #emg_diff = eeg_df['EMG2'] - eeg_df['EMG1']
            emg_diff = eeg_df["EMG1"]
            # Perform prediction and plot spectrogram for the EEG channel
            hypno, proba = predict_electrode(eeg=eeg, emg=emg_diff, epoch_sec=epoch_sec)
            #plot_spectrogram(eeg, hypno)
            
            # Store hypno and proba in the results dictionary under the column key
            results[column] = {"hypno": hypno, "proba": proba}

    hypno_predictions_df = aggregate_hypno_predictions(results)
    max_probabilities_df = get_max_probabilities(results)

    mfv = get_most_frequent_value(hypno_predictions_df)
    consensus = consensus_prediction(hypno_predictions_df, max_probabilities_df)

    output_df = pd.DataFrame({'consensus': consensus, 'mfv' : mfv}).apply(yasa.hypno_int_to_str)

    # Saving data 
    output_fn = bids_naming(session_folder, animal_id, session, 'consensus_mfv_predictions.csv.gz')
    output_df.to_csv(output_fn, index=False)
    console.success(f"Wrote consensus predictions to {output_fn}")
    hypno_fn = bids_naming(session_folder, animal_id, session, 'hypno_predictions.csv.gz')
    hypno_predictions_df.to_csv(hypno_fn, index=False)
    console.success(f"Wrote predictions to {hypno_fn}")
    proba_fn = bids_naming(session_folder, animal_id, session, 'max_probabilities.csv.gz')
    max_probabilities_df.to_csv(proba_fn, index=False)
    console.success(f"Wrote probas to {proba_fn}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--epoch_sec", type=float, required=True, help="Epoch for sleep predictions in seconds. Ideally, it matches the classifier epoch_sec")
  args = parser.parse_args()
  epoch_sec = args.epoch_sec
  # Caution hard-coded!
  sf = 100
  console.warn(f'sf = 100 is hardcoded!')
  console.log(f'Running `predict.py` with sf={sf} and epoch_sec={args.epoch_sec}')
  predict_and_save(epoch_sec)