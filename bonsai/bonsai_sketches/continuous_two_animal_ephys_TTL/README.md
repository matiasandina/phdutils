This folder contains code to handle the EEG/EMG and video data acquisition with bonsai.

It contains two groups of files.

### Data Acquisition

* `continuous_two_animal_ephys_TTL.bonsai` <- the bonsai sketch itself
* `rhd2000.bit` <- needed for open ephys DAQ
* `rhd200_usb3.bit` <- needed for open ephys DAQ

These files will acquire from two boxes continuously chopping files with a certain period defined in a `Timer` node.
It has the option to manually trigger a camera recording and TTL synchronization using keyboard input.

### File Handling

The bonsai sketch generates a ton of individual files and handling paths and filenaming with a convention inside bonsai is problematic. I am somewhat following the [BIDS](https://bids-standard.github.io/bids-starter-kit/index.html) format. The files created here go into a `database_path`

* `move_files.py` <- movement of files itself
* `config.yaml` <- metadata to name the files properly and contains the paths where to look for things
