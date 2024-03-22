import os
import argparse
import datetime
from py_console import console
from bonsai_dat_to_npy_eeg import *
from predict import *

def run_pipeline(start_date=None, animal_id=None):
    base_folder = f"/synology-nas/MLA/beelink1/{animal_id}"

    # Check if base folder exists
    if not os.path.exists(base_folder) or not os.path.isdir(base_folder):
        console.error(f"Error: The base folder '{base_folder}' does not exist or is not a directory.")
        console.error("Make sure the path is correct and/or the NAS is mounted properly to the '/synology-nas' directory.")
        return

    # Get a sorted list of folders
    folders = sorted(entry for entry in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, entry)))
    # only keep the ones with date pattern
    folders = [folder for folder in folders if is_valid_date(folder)]

    # Filter folders based on the start date
    if start_date:
        start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        folders = [folder for folder in folders if datetime.datetime.strptime(folder, "%Y-%m-%d") >= start_datetime]

    # TODO: We assume one config per animal at base_folder....this might change soon
    config = read_config(base_folder)
    assert config['down_freq_hz'] is not None, "No down_freq_hz in config. Exiting function"
    assert config["aq_freq_hz"] > config["down_freq_hz"], f"{config['aq_freq_hz']} must be greater than {config['down_freq_hz']}"

    # Loop over the folders
    for folder in folders:
        ephys_folder_path = os.path.join(base_folder, folder, "eeg")
        ttl_folder_path = os.path.join(base_folder, folder, "ttl")
        # Handle file finding
        console.info(f"Working on {ephys_folder_path}.")
        console.info("Finding _eeg.bin files")
        file_list = list_files(ephys_folder_path, pattern="_eeg.bin", full_names=True)

        #TODO: ADD the creation and checking of parameter dicts
        # THIS WOULD HELP US SKIP STEPS IF PREVIOUSLY COMPUTED

        # Run the Python script with the appropriate arguments
        downsampled_eegs, nsamples = filter_down_bonsai_eeg(config, file_list, output_folder= ephys_folder_path)


        # Optionally, you can add any post-processing steps here
        #if os.path.isdir(ttl_folder_path):
        #    console.info("Found ttl folder, there's an interactive folder selection in this step!")
        #    # do the alignment, this is interactive to select the specific TDT data though
        #    command = f"python3 02-bonsai_ttl_alignment.py --session_folder {os.path.dirname(ttl_folder_path)} --config_folder {base_folder}"
        #    os.system(command)
        # do the prediction
        
        # TODO: this returns nothing for now
        run_and_save_predictions(animal_id, date=folder, epoch_sec = 2.5, eeg_data_dict = downsampled_eegs, config=config, display=False)
        # Print a newline for separation between iterations
        console.log(f"Finished folder {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", help="Starting date for batch processing (format: YYYY-MM-DD)")
    parser.add_argument("--animal_id", required=True, help="Animal ID for constructing the base path")
    args = parser.parse_args()

    run_pipeline(args.start_date, args.animal_id)