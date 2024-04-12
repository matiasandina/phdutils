import py_compile
import yasa
from staging import SleepStaging
import mne
from mne.io import RawArray
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl
from rlist_files import list_files
from py_console import console
from utils import *
import argparse
from sklearn.preprocessing import RobustScaler, robust_scale, minmax_scale

def list_session_dates(base_folder, start_date=None):
    # helper to list session dates
    if not os.path.exists(base_folder) or not os.path.isdir(base_folder):
        console.error(f"Error: The base folder '{base_folder}' does not exist or is not a directory.")
        return []

    folders = sorted([entry for entry in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, entry)) and is_valid_date(entry)])

    if start_date:
        folders = [folder for folder in folders if datetime.datetime.strptime(folder, "%Y-%m-%d").date() >= start_date]

    return folders

def predict_electrode(eeg, emg, sf, epoch_sec = 2.5):
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

def plot_spectrogram(eeg, hypno, sf, epoch_sec = 2.5):
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

def check_path_exists(base_folder, date):
  # Check if base folder exists
  if not os.path.exists(base_folder) or not os.path.isdir(base_folder):
      console.error(f"Error: The base folder '{base_folder}' does not exist or is not a directory.")
      console.error("Make sure the path is correct and/or the NAS is mounted properly to the '/synology-nas' directory.")
      return None, None
  session_folder = os.path.join(base_folder, date)
  eeg_folder = os.path.join(session_folder, "eeg")
  # Check if eeg folder exists
  if not os.path.exists(eeg_folder) or not os.path.isdir(eeg_folder):
      console.error(f"Error: The base folder '{eeg_folder}' does not exist or is not a directory.")
      console.error(f"Date was {date}, is this correct?, check `ls {eeg_folder}`")
      return None, None

  return session_folder, eeg_folder

def display_electrodes(results):
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
  plt.show(block = False)
  return

def clip_quantiles_startswith(df, prefix="EEG", upper_quantile=0.999, lower_quantile=0.001):
  # Identify columns that start with the specified prefix
  columns_to_clip = [col for col in df.columns if col.startswith(prefix)]
  
  # Iterate over these columns and apply the clipping
  for column in columns_to_clip:
    upper_bound = df[column].quantile(upper_quantile)
    lower_bound = df[column].quantile(lower_quantile)
    df = df.with_columns(
        pl.when(df[column] > upper_bound)
        .then(upper_bound)
        .when(df[column] < lower_bound)
        .then(lower_bound)
        .otherwise(df[column])
        .alias(column)
    )
  return df

def normalize_eegs(data, method="robust", keep_dims = True):
    assert method in ['robust', 'minmax'], f"Error: Scaling method must be either 'robust' (default) or 'minmax', received {method}"
    if method=="robust":
        scaled = data.select(pl.selectors.starts_with("EEG").map_batches(lambda x: pl.Series(robust_scale(x, with_centering=True, with_scaling=True, unit_variance = False))))
    if method=="minmax":
        scaled = data.select(pl.selectors.starts_with("EEG").map_batches(lambda x: pl.Series(minmax_scale(x))))
    if keep_dims:
        return scaled.hstack(data.select(~pl.selectors.starts_with("EEG")))
    else:
        return scaled
        
def normalize_data(data, method="robust"):
    assert method in ['robust', 'minmax'], f"Error: Scaling method must be either 'robust' (default) or 'minmax', received {method}"
    if method=="robust":
        scaled = data.select(pl.all().map_batches(lambda x: pl.Series(robust_scale(x, with_centering=True, with_scaling=True, unit_variance = False))))
    if method=="minmax":
        scaled = data.select(pl.all().map_batches(lambda x: pl.Series(minmax_scale(x))))
    return scaled

def process_eeg(eeg_df, sf, epoch_sec, robust_scale=True, display=False):
  # Artifact detection
  lower_quant = 0.01
  upper_quant = 0.99
  console.info(f"Performing artifact clipping with lower:{lower_quant} and upper:{upper_quant}")
  eeg_df = clip_quantiles_startswith(eeg_df,  prefix="EEG", lower_quantile=lower_quant, upper_quantile=upper_quant)

  # Scale the dataframe using robust scaler
  # using lambda x with pl.Series might be slower ?
  # ColumnTransformer from sklearn, but also update to 1.4 so that we get output="polars"
  if robust_scale:
    console.info("Scaling columns using robust scaler. Check Before and After!!")
    console.log(f"Original data is of shape {eeg_df.shape}")
    print("=" * os.get_terminal_size().columns)
    print(eeg_df.describe())
    #only scale eegs
    #eeg_df = normalize_eegs(eeg_df, method = "robust")
    # if model was trained with scaled emgs
    # issue is we clip quantiles for the eeg only, how robust the scaling? 
    eeg_df = normalize_data(eeg_df, method = "robust")
    console.log(f"Scaled data is of shape {eeg_df.shape}")
    print("=" * os.get_terminal_size().columns)
    console.warn("Check Effective scaling below!")
    print("=" * os.get_terminal_size().columns)
    print(eeg_df.describe())

  results = {}
  for column in eeg_df.columns:
    if column.startswith('EEG'):
        eeg = eeg_df[column].to_numpy()
        # Calculate EMG difference for each EEG channel
        emg_diff = eeg_df["EMG1"]
        #emg_diff = eeg_df['EMG2'] - eeg_df['EMG1']
        # Perform prediction and plot spectrogram for the EEG channel
        hypno, proba = predict_electrode(eeg=eeg, emg=emg_diff, sf = sf, epoch_sec=epoch_sec)
        #plot_spectrogram(eeg, hypno)
        # Store hypno and proba in the results dictionary under the column key
        results[column] = {"hypno": hypno, "proba": proba}

  if display:
    display_electrodes(results)

  hypno_predictions_df = aggregate_hypno_predictions(results)
  max_probabilities_df = get_max_probabilities(results)
  # We will do consensus and most frequent value using all electrodes
  # TODO: This is likely a waste of time/effort
  mfv = get_most_frequent_value(hypno_predictions_df)
  consensus = consensus_prediction(hypno_predictions_df, max_probabilities_df)
  # agregate into an output dataframe
  consensus_df = pd.DataFrame({'consensus': consensus, 'mfv' : mfv}).apply(yasa.hypno_int_to_str)
  return {'hypno_predictions_df': hypno_predictions_df, 'max_probabilities_df': max_probabilities_df, 'consensus_df': consensus_df}


def save_predictions(data_dict, saving_folder, animal_id, session_id):
    for key, df in data_dict.items():
        # Generate a filename for each DataFrame based on its key
        filename = bids_naming(saving_folder, animal_id, session_id, f'{key}.csv.gz')
        df.to_csv(filename, index=False)
        console.success(f"Wrote {key} data to {filename}")

def is_dataframe(df):
  return isinstance(df, pl.dataframe.frame.DataFrame) or isinstance(df, pd.DataFrame)

def run_and_save_predictions(animal_id, date, epoch_sec, eeg_data_dict = None, config=None, robust_scale=True, display=False):
  base_folder = os.path.join("/synology-nas/MLA/beelink1", animal_id)
  # coerce date back to yyyy-mm-dd as character
  date = str(date)
  session_folder, eeg_folder = check_path_exists(base_folder, date)
  if config is not None:
    sf = config['down_freq_hz']
  if session_folder is None or eeg_folder is None:
      console.error(f"Path check under {base_folder} failed. Exiting the function.")
      return  # Exit the function if path check fails
  # let's save predictions under session_folder/sleep
  saving_folder = os.path.join(session_folder, "sleep")
  if not os.path.exists(saving_folder):
    console.info(f"Creating directory {saving_folder} to save sleep predictions")
    os.makedirs(saving_folder)
  # Deal with eeg data or paths to eeg_files
  if isinstance(eeg_data_dict, dict) and all(is_dataframe(df) for df in eeg_data_dict.values()):
    console.info("Received a dict of dataframes as input")
    for eeg_file, df in eeg_data_dict.items():
      session_id = parse_bids_session(os.path.basename(eeg_file))
      console.log(f"session_id: {session_id}. Predicting electrodes in file {os.path.basename(eeg_file)}.")
      output_dict = process_eeg(df, sf, epoch_sec, display)
      save_predictions(output_dict, saving_folder, animal_id, session_id)
  else: 
    # Find downsampled eeg files and trigger prediction for each
    # We should have only one downsampling, but this will match all downsampling factors
    # Not addressing that concern now
    console.log("Finding downsampled EEG file(s) for prediction")
    eeg_files = list_files(eeg_folder, pattern = "*desc-down*csv.gz", full_names = True)
    if not eeg_files:
      console.error("No EEG files found for processing.")
      return
    # Trigger prediction
    for eeg_file in eeg_files:
      session_id = parse_bids_session(os.path.basename(eeg_file))
      eeg_df = pl.read_csv(eeg_file)
      console.log(f"session_id: {session_id}. Predicting electrodes in file {os.path.basename(eeg_file)}.")
      output_dict = process_eeg(eeg_df, sf, epoch_sec, robust_scale, display)
      # Save the data 
      save_predictions(output_dict, saving_folder, animal_id, session_id)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--animal_id", required=True, help="Animal ID for constructing the base path. /path_to_storage/animal_id")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--date", type=datetime.date.fromisoformat, help="Specific date to be analyzed in YYYY-MM-DD format. (/path_to_storage/animal_id/date/eeg/)")
  group.add_argument("--start_date", type=datetime.date.fromisoformat, help="Start date for processing sessions (inclusive). Format: YYYY-MM-DD. All sessions from this date onwards will be considered.")
  parser.add_argument('--config_folder', help='Path to the config folder')
  parser.add_argument("--epoch_sec", type=float, required=True, help="Epoch for sleep predictions in seconds. Ideally, it matches the classifier epoch_sec")
  args = parser.parse_args()
  config = read_config(args.config_folder)
  sf = config['down_freq_hz']
  console.log(f'Running `predict.py` with sf={sf} and epoch_sec={args.epoch_sec}')

  base_folder = os.path.join("/synology-nas/MLA/beelink1", args.animal_id)

  if args.date:
      dates = [args.date]
  else:  # args.start_date is given
      dates = list_session_dates(base_folder, args.start_date)

  for date in dates:
      console.log(f'Processing for date: {date}')
      run_and_save_predictions(animal_id=args.animal_id, date=date, epoch_sec=args.epoch_sec, config=config)
