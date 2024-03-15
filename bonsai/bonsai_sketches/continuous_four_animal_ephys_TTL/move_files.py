import os
import time
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import re
from datetime import datetime, timedelta
from yaspin import yaspin
from yaspin.spinners import Spinners
from rlist_files import list_files

# Read the configuration from the YAML file
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract the configuration values for period, database path, and directory to watch
period = config["period"]
period_parts = period.split(":")
sleep_time = int(period_parts[0]) * 3600 + int(period_parts[1]) * 60 + int(period_parts[2])

database_path = config["database_path"]
directory_to_watch = config["directory_to_watch"]

# Dynamically extract box configuration
# Assuming you have box1, box2, box3, box4 in your config, adjust if you have more
num_animals 4
# Creates {'box1': id1, 'box2': id2, ...}
# config.get() will default to False if the box key.
# If not running all 4 boxes, you can leave them empty and if config.get() will be evaluated as False
# and boxes will not be included 
box_config = {f"box{i}": config.get(f"box{i}") for i in range(1, num_animals + 1) if config.get(f"box{i}", False)}  

# Create the base directories for each box
box_directories = {}  # To store directory paths for each box
for box, box_id in box_config.items():
    box_directory = os.path.join(database_path, box_id)
    box_directories[box] = box_directory

# Dynamically handle directory existence check and renaming if needed
for box, directory in box_directories.items():
    if os.path.exists(directory):
        new_id = get_new_dir_name(box_config[box])  # get_new_dir_name needs to handle box ids dynamically
        box_config[box] = new_id  # Update the box id in the configuration if it's renamed
        box_directories[box] = os.path.join(database_path, new_id)  # Update the directory path

# Function to get new directory name if the id already exists (e.g., you forgot to update the config.yaml)
# It's basically a glorified way to check that you are writing in the proper place
def get_new_dir_name(box_id):
    new_id = input(f"The directory for `{box_id}` already exists.\n>> Please enter a new id or press enter to use the old one: ")
    return new_id if new_id else box_id

def rename_file(destination_folder, file_path, box_id, timestamp):
    """
    Renames a file following the specified naming convention and moves it to the appropriate folder.
    """
    file_name = os.path.basename(file_path)
    base_name, extension = os.path.splitext(file_name)

    # We will get a datetime here and we have to parse back again
    timestamp = timestamp.strftime("%Y-%m-%dT%H_%M_%S")
    # Remove "_" and "-" from the timestamp
    timestamp = re.sub(r"[-_]", "", timestamp)

    # Match statement to determine the new file name and destination folder
    match file_name:
        case f if f.startswith("box") and "eegdata" in base_name:
            new_file_name = f"sub-{box_id}_ses-{timestamp}_eeg{extension}"
            destination_folder = os.path.join(destination_folder, "eeg")
        case f if f.startswith("box") and "vid" in base_name:
            new_file_name = f"sub-{box_id}_ses-{timestamp}_video{extension}"
            destination_folder = os.path.join(destination_folder, "video")
        case f if f.startswith("box"):
            new_file_name = f"sub-{box_id}_ses-{timestamp}_{base_name}{extension}"
        case f if f.startswith("vid_timestamp"):
            new_file_name = f"sub-{box_id}_ses-{timestamp}_timestamp{extension}"
            destination_folder = os.path.join(destination_folder, "video")
        case f if f.startswith("ttl_in_state"):
            new_file_name = f"sub-{box_id}_ses-{timestamp}_ttl_in{extension}"
            destination_folder = os.path.join(destination_folder, "ttl")
        case _:
            print(f"Skipping {file_name} as it doesn't match any file type.")
            return

    # Construct the new file path
    # Destination folder already has the date YYYY-mm-dd 
    new_file_path = os.path.join(destination_folder, new_file_name)

    # Create the necessary directories
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Rename and move the file
    os.rename(file_path, new_file_path)
    spinner.color = "green"
    spinner.write(f"✔ Renamed and moved {file_name} to {new_file_path}")
    spinner.color = "white"

def process_file(file_path):
    """
    Processes a file by determining its box and moving it to the appropriate destination folder,
    taking into account the configured period and avoiding moving files that are being written to.
    """
    file_name = os.path.basename(file_path)
    box_id, box_directory = None, None
    
    # Dynamically determine the box_id and directory based on the file name
    for box, id in box_config.items():
        if box in file_name:
            box_id = id
            box_directory = box_directories[box]
            break
    
    if not box_id or not box_directory:
        spinner.fail(f"> Failed to determine box from {file_name}. Skipping the file.")
        return

    # Extract the timestamp using regular expressions
    pattern = r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}"
    match = re.search(pattern, file_name)
    if not match:
        spinner.fail(f"> Failed to extract timestamp from {file_name}. Skipping the file.")
        return

    # Get the matched timestamp
    timestamp = match.group()

    # Convert the timestamp to datetime format
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H_%M_%S")

    # Calculate the current time
    current_time = datetime.now()

    # Parse the period as a timedelta object
    period_timedelta = timedelta(hours=int(period_parts[0]), minutes=int(period_parts[1]), seconds=int(period_parts[2]))

    # Calculate the threshold time
    threshold_time = current_time - (3 * period_timedelta)

    # Check if the file is still being written to
    if timestamp >= threshold_time:
        spinner.ok(f"> Skipping {file_name} as it might still be written to.")
        return

    # Check if the file is empty
    if os.path.getsize(file_path) == 0:
        spinner.ok(f"Removing empty file: {file_name}")
        os.remove(file_path)
        return

    # Determine the destination folder based on the box and date
    destination_folder = os.path.join(box_directory, timestamp.strftime("%Y-%m-%d"))

    # Move the file to the destination folder
    rename_file(destination_folder, file_path, box_id, timestamp)


# If we needed an on_creation() listener we could use this
# With bonsai, it might turn impractical
# class FileHandler(FileSystemEventHandler):
#     def __init__(self):
#         super().__init__()
#         self.processing = False

#     def on_created(self, event):
#         if not event.is_directory and not self.processing:
#             self.processing = True
#             process_file(event.src_path)
#             self.processing = False

#     def on_modified(self, event):
#         if not event.is_directory and not self.processing:
#             self.processing = True
#             process_file(event.src_path)
#             self.processing = False

# # Create the observer and event handler
# observer = Observer()
# event_handler = FileHandler()

# # Schedule the observer to watch the target directory
# observer.schedule(event_handler, directory_to_watch, recursive=False)

# # Start the observer in the background
# observer.start()

#spinner = yaspin()
#spinner.start()


# Try to keep the script running
#try:
#    while observer.is_alive():
#        observer.join(1)
#except KeyboardInterrupt:
#    print("Received keyboard interrupt, stopping observer.")
#    spinner.stop()
#    observer.stop()

spinner = yaspin()

try:
    while True:
        for file in list_files(path = directory_to_watch, full_names = True):
            spinner.start()
            spinner.write(f"checking {file}")
            process_file(file)
        spinner.start()
        spinner.spinner = Spinners.moon
        spinner.text = f"Waiting for {sleep_time} seconds"
        time.sleep(sleep_time)
        spinner.spinner = Spinners.toggle
except KeyboardInterrupt:
    spinner.fail("Received keyboard interrupt, stopping.")
