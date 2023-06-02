This folder contains code to handle the EEG/EMG and video data acquisition with bonsai.

It contains two groups of files.

## Bonsai

### Data Acquisition

* `continuous_two_animal_ephys_TTL.bonsai` <- the bonsai sketch itself
* `rhd2000.bit` <- needed for open ephys DAQ
* `rhd200_usb3.bit` <- needed for open ephys DAQ

These files will acquire from two boxes continuously chopping files with a certain period defined in a `Timer` node.
It has the option to manually trigger a camera recording and TTL synchronization using keyboard input.

### Data Encoding

The bonsai file will save many `.bin` files stored as ColumnMajor. In order to read them properly, we have to use the proper encoding and know the expected columns x timepoints.

* `eegdata` will be number of channels as in the `selectChannels` node, and encoded to be read with `np.float32`
* `vid_timestamp` will be encoded as ColumnMajor in this order: cam1_frame_number, cam2_frame_number, AmplifierCount encoded as `np.int32`. 
* `ttl_in_state` will be each of the 8 channels stored as ColumnMajor, encoded as `np.int8`

### File Handling in Python

The bonsai sketch generates a ton of individual files and handling paths and filenaming with a convention inside bonsai is problematic. I am somewhat following the [BIDS](https://bids-standard.github.io/bids-starter-kit/index.html) format. The files created here go into a `database_path`

* `move_files.py` <- movement of files itself
* `config.yaml` <- metadata to name the files properly and contains the paths where to look for things

Currently, these files are made to handle 2 boxes and either marked with 'box1' or 'box2'. Files not marked will default to 'box2', this might create errors or produce unexpected behavior. Be aware of this !