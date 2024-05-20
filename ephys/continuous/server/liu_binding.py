import os
from py_console import console
import argparse
from rlist_files import list_files
import datetime
import polars as pl

def parse_bids_session(string: str):
    return os.path.basename(string).split("_")[1].replace("ses-", "")

def merge_sessions(base_folder, animal_id, cutoff_time):
    files = list_files(path=base_folder, pattern='desc-down*_eeg.csv.gz', recursive=True, full_names=True)
    console.info("Found these files")
    print("\n".join(files))
    cutoff_time = datetime.datetime.strptime(cutoff_time, '%H:%M:%S').time()

    # Group files by day and session
    sessions = {}
    for file in files:
        session_datetime_str = parse_bids_session(file)
        session_datetime = datetime.datetime.strptime(session_datetime_str, '%Y%m%dT%H%M%S')
        console.info(f"Session datetime extracted: {session_datetime}")

        # Determine if this file belongs to the current day session or the next day's session
        session_date = session_datetime.date()
        if session_datetime.time() < cutoff_time:
            session_date -= datetime.timedelta(days=1)
            console.info(f"File {file} assigned to previous day's session")

        if session_date not in sessions:
            sessions[session_date] = []
        sessions[session_date].append(file)

    # Process each session
    for idx, (session_date, files) in enumerate(sorted(sessions.items()), 1):
        console.log("Reading the following files. This might take a while!")
        console.log(f"{files}")
        dfs = [pl.read_csv(file) for file in files]
        row_nums = list(map(lambda x: x.height, dfs))
        for item in zip(files, row_nums):
            print(f"(File, n_rows) {item}")
        console.success(f"Concat in progress")
        df_concat = pl.concat(dfs)
        assert sum(row_nums) == df_concat.height, "Concat resulted in loss of data, aborting!!!"
        # Determine session datetime for naming from the first file in the sorted list
        first_session_datetime = parse_bids_session(sorted(files)[0])
        output_dir = os.path.join(base_folder, f'day0{idx}')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'sub-{animal_id}_ses-{first_session_datetime}.csv.gz')
        df_concat.to_pandas().to_csv(output_file, index=False)
        console.success(f"Saved concatenated session to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal_id", required=True, help="Animal ID for constructing the base path")
    parser.add_argument("--base_folder", required=False, help="Full path of base folder if not using default hard-coded one", default=None)
    parser.add_argument("--cutoff_time", required=True, help="HH:MM:SS to use as cutoff for grouping sessions into experimental days instead of calendar dates")
    args = parser.parse_args()
    base_folder = args.base_folder if args.base_folder else os.path.join("/synology-nas/MLA/LY", args.animal_id)
    console.info(f"Using path: {base_folder}")

    merge_sessions(base_folder, args.animal_id, args.cutoff_time)