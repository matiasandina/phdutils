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


# This is what has to run
# TODO: fix the path
def run_and_save_predictions(animal_id, date, display=False ):
  base_folder = os.path.join("/synology-nas/MLA/beelink1", animal_id)
  # coerce date back to yyyy-mm-dd as character
  date = str(date)
  # Check if base folder exists
  if not os.path.exists(base_folder) or not os.path.isdir(base_folder):
      console.error(f"Error: The base folder '{base_folder}' does not exist or is not a directory.")
      console.error("Make sure the path is correct and/or the NAS is mounted properly to the '/synology-nas' directory.")
      return
  session_folder = os.path.join(base_folder, date)
  eeg_folder = os.path.join(session_folder, "eeg")
  # Check if eeg folder exists
  if not os.path.exists(eeg_folder) or not os.path.isdir(eeg_folder):
      console.error(f"Error: The base folder '{eeg_folder}' does not exist or is not a directory.")
      console.error(f"Date was {date}, is this correct?, check `ls {eeg_folder}`")
      return

  console.log("Finding downsampled EEG file for prediction")
  eeg_file = list_files(eeg_folder, pattern = "*desc-down*csv.gz", full_names = True)
  # This might happen if data is chunked I believe
  if len(eeg_file) > 1:
    console.error("More than one match. Check eeg_file pattern")
    print(eeg_file)
    return
  console.log(f"Working with {eeg_file[0]}")
  eeg_df = pl.read_csv(eeg_file[0])
  results = {}
  for column in eeg_df.columns:
    if column.startswith('EEG'):
        eeg = eeg_df[column].to_numpy()
        # Calculate EMG difference for each EEG channel
        #emg_diff = eeg_df["EMG1"]
        emg_diff = eeg_df['EMG2'] - eeg_df['EMG1']
        # Perform prediction and plot spectrogram for the EEG channel
        hypno, proba = predict_electrode(eeg, emg_diff)
        #plot_spectrogram(eeg, hypno)
        # Store hypno and proba in the results dictionary under the column key
        results[column] = {"hypno": hypno, "proba": proba}

  if display:
    # this becomes really hard to read with long recordings
    # electrodes lineplot with vertical spacing
    # Get the list of electrode keys from the results dictionary
    electrodes = list(results.keys())

    # Determine the number of electrodes
    num_electrodes = len(electrodes)

    # Create a figure and axes for the plot
    fig, ax = plt.subplots()

    # Plot each electrode separately with vertical spacing
    for i, electrode in enumerate(electrodes):
        proba = results[electrode]['proba'].max(axis=1)
        ax.plot(proba + i, label=electrode)
        # Calculate mean and standard deviation
        mean_proba = proba.mean()
        std_proba = proba.std()
        # Add mean + SD as text next to the trace
        ax.text(len(proba), i + 1, f'{mean_proba:.3f} ± {std_proba:.3f}', va='center')

    # Set labels and title
    ax.set_ylabel('Electrode')
    ax.set_title('Sleep Stage Prediction Probability Between Electrodes')

    # Set y-axis ticks and labels
    ax.set_yticks(np.arange(num_electrodes) + 1)
    ax.set_yticklabels(electrodes)

    # Set x-axis label and tick positions
    ax.set_xlabel('Epoch')
    ax.set_xticks(np.arange(0, len(proba), step=200))

    # Adjust the x-range of the plot
    ax.set_xlim(0, len(proba) + 250)

    # Display the plot
    plt.show()


  hypno_predictions_df = aggregate_hypno_predictions(results)
  max_probabilities_df = get_max_probabilities(results)

  mfv = get_most_frequent_value(hypno_predictions_df)
  consensus = consensus_prediction(hypno_predictions_df, max_probabilities_df)

  output_df = pd.DataFrame({'consensus': consensus, 'mfv' : mfv}).apply(yasa.hypno_int_to_str)

  # Saving data 
  output_fn = bids_naming(session_folder, animal_id, date, 'consensus_mfv_predictions.csv.gz')
  output_df.to_csv(output_fn, index=False)
  console.success(f"Wrote consensus predictions to {output_fn}")
  hypno_fn = bids_naming(session_folder, animal_id, date, 'hypno_predictions.csv.gz')
  hypno_predictions_df.to_csv(hypno_fn, index=False)
  console.success(f"Wrote predictions to {hypno_fn}")
  proba_fn = bids_naming(session_folder, animal_id, date, 'max_probabilities.csv.gz')
  max_probabilities_df.to_csv(proba_fn, index=False)
  console.success(f"Wrote probas to {proba_fn}")

def bids_naming(session_folder, subject_id, session_date, filename):
    session_date = session_date.replace("-", "")
    return(os.path.join(session_folder, f"sub-{subject_id}_ses-{session_date}_{filename}"))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--animal_id", required=True, help="Animal ID for constructing the base path. /path_to_storage/animal_id")
  parser.add_argument("--date", required=True, 
    type=datetime.date.fromisoformat,
    help="Date that wants to be analized yyyy-mm-dd, used to construct folder path (/path_to_storage/animal_id/date/eeg/)")
  parser.add_argument('--config_folder', help='Path to the config folder')
  args = parser.parse_args()
  config = read_config(args.config_folder)
  sf = config['down_freq_hz']
  console.log(f'Running `predict.py` with sf={sf}')
  run_and_save_predictions(args.animal_id, args.date)  
