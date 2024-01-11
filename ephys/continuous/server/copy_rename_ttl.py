import argparse
import shutil
from utils import list_files
from py_console import console
import os
from pathlib import Path

def rename_ttl_files(copy_from, copy_to, force_copy):
    for file, new_file in zip(copy_from, copy_to):
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        if os.path.isfile(new_file):
            console.warn(f"{new_file} exist in destination")
            if force_copy:
                console.info(f"Overwriting exis")
                shutil.copy(file, new_file)
            else:
                console.info(f"Skipping file. To force copy use `force_copy = True`")
        else:
            console.info(f"Copying {file} into {new_file}")
            shutil.copy(file, new_file)

def copy_rename_ttl(nas_path, from_id, to_id, force_copy):
    # use pathlib to list the directories inside joining nas_path and from_id
    source_dir = Path(nas_path) / from_id
    # these are the recorded sessions for each date
    directories = [directory for directory in source_dir.iterdir() if directory.is_dir()]
    # Check if a subdirectory named "ttl" exists
    ttl_dirs = [directory / "ttl" for directory in directories if (directory / "ttl").is_dir()]

    # Iterate over the files and copy the ones with the pattern "ttl"
    for dir in ttl_dirs:
        files = list_files(dir, full_names=True, pattern = "ttl")
        new_files = [file.replace(from_id, to_id) for file in list_files(str(dir), full_names=True)]
        rename_ttl_files(files, new_files, force_copy)

if __name__ == "__main__":
    # hardcoded nas path
    nas_path = "/synology-nas/MLA/beelink1/"
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="This program copies TTL files from one animal's session subfolders into the animal that was paired with it during acquisition. If animal2 ran with animal1 and animal2 has the original ttl files, we can do `python3 copy_rename_ttl.py --from_id 'animal2' --to_id 'animal1'`")
    parser.add_argument("--from_id", help="Id of animal that contains the original copy of ttls")
    parser.add_argument("--to_id", help="Id of animal ran with `from_id`")
    parser.add_argument("--force_copy", help="If files already exist, we will not copy unless forced", default=False)
    args = parser.parse_args()
    # print information about the call using console.info
    console.info(f"Nas path is {nas_path}")
    console.info(f"Copying and renaming TTL files from {args.from_id} to {args.to_id}")
    # Call the copy_rename_ttl function with the provided directory paths
    copy_rename_ttl(nas_path = nas_path, from_id = args.from_id, to_id = args.to_id, force_copy = args.force_copy)
