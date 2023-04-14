# UI
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import QSysInfo, QTime, QTimer

# Functions
import tdt
import numpy as np
import random
import pandas as pd
import random

## Experiment PARAMS #####

# Set the experiment
sExperiment = "PhotoOptoRandTrialsTTL2Box"
exp_duration_hours = 0.1 #4
exp_duration_seconds = exp_duration_hours * 3600
# we can put some large number here, anyway the amount of true trials is determined by the exp duration
n_trials = 500

# This goes into Period1
trial_period_ms = 120 * 1000
# This goes into Width1
trial_width_ms = trial_period_ms - 0.1 # can't be 100% duty cycle
# This goes into Period2
stim_period_ms = 5 * 1000
# This goes into Width2 <- we want to change this width to increase the duty cycle and have less time off stim
stim_width_ms = 1 * 1000 
# This goes into Period3 <- we want to change this to change frequency of stimulation
light_period_ms = 50
# This goes into Width3 <- we want to keep this one here fixed
light_width_ms = 10

# Light params ######
frequencies_hz = [1, 5, 10, 20]
# right now, this is what I'm using
# weights =  [0, 0, 0, 1]
weights = [0.25, 0.25, 0.25, 0.25]
stim_duty_cycles = [0.2, 0.5, 0.75, 0.999]
# Right now, this is what I am using
#stim_weights = [1, 0, 0, 0]
stim_weights = [0.25, 0.25, 0.25, 0.25]

def generate_stim_trials(stim_duty_cycles, weights, stim_period_ms, n_trials):
    trials = pd.DataFrame()
    for i in range(n_trials):
        trial_params = generate_stim_trial(stim_duty_cycles, weights, stim_period_ms)
        trials = pd.concat([trials, pd.DataFrame(trial_params)], ignore_index=True)
    trials['trial_n'] = np.arange(1, n_trials + 1)
    return trials

def generate_stim_trial(stim_duty_cycles, weights, stim_period_ms):
    if len(stim_duty_cycles) != len(weights):
        raise ValueError("The length of stim_duty_cycle and weights must be the same.")
    if abs(sum(weights) - 1) > 1e-6:
        raise ValueError("The sum of weights must be equal to 1.")
        
    # randomly select a frequency based on the weights
    duty_cycle = random.choices(stim_duty_cycles, weights=weights)[0]
    stim_width_ms = stim_period_ms * duty_cycle
    stim_off_ms = stim_period_ms - stim_width_ms
    trial_params = pd.DataFrame({
        "stim_duty_cycle": [duty_cycle],
        "stim_period_ms": [stim_period_ms],
        "stim_width_ms": [stim_width_ms],
        "stim_off_ms": [stim_off_ms],
    })
    return trial_params


def generate_light_trial(frequencies_hz, weights, light_width_ms):
    if len(frequencies_hz) != len(weights):
        raise ValueError("The length of frequencies_hz and weights must be the same.")
    if abs(sum(weights) - 1) > 1e-6:
        raise ValueError("The sum of weights must be equal to 1.")
        
    # randomly select a frequency based on the weights
    freq = random.choices(frequencies_hz, weights=weights)[0]
    
    # calculate the light period and off time
    light_period_ms = 1000 / freq
    light_off_ms = light_period_ms - light_width_ms
    
    # calculate the duty cycle
    duty_cycle = light_width_ms / light_period_ms
    
    # return the trial parameters as a pandas DataFrame
    trial_params = pd.DataFrame({
        "frequency_hz": [freq],
        "duty_cycle": [duty_cycle],
        "light_period_ms": [light_period_ms],
        "light_width_ms": [light_width_ms],
        "light_off_ms": [light_off_ms]
    })
    
    return trial_params


def generate_light_trials(frequencies_hz, weights, light_width_ms, n_trials):
    trials = pd.DataFrame()
    for i in range(n_trials):
        trial_params = generate_light_trial(frequencies_hz, weights, light_width_ms)
        trials = pd.concat([trials, trial_params])
    # reset the index so it's not 0 for each row
    trials.reset_index(drop=True, inplace=True)
    trials['trial_n'] = np.arange(1, n_trials+1)
    return trials


class ExperimentGUI(QWidget):
    def __init__(self, syn, trials_df, train_names, train_labels):
        super().__init__()
        self.syn = syn
        self.trials_df = trials_df
        self.exp_duration_seconds = exp_duration_seconds
        self.train_names = ['Train1', 'Train2']

        self.init_ui()

        # initialize timer and elapsed time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_time)
        self.start_time = QTime.currentTime()
        self.elapsed_time = QTime(0, 0, 0)
        # start timer to track elapsed time
        self.timer.start(1000)

        # set trial label
        self.trial_label.setText('Trial 1')

    def init_ui(self):
        # set up GUI elements
        self.setWindowTitle('Experiment GUI')
        self.setGeometry(100, 100, 500, 200)

        self.train_buttons = []
        for i, train_name in enumerate(self.train_names):
            button = QPushButton(train_name, self)
            button.setGeometry(50 + i*150, 50, 100, 50)
            button.clicked.connect(self.on_train_button_clicked)
            self.train_buttons.append(button)

        self.trial_label = QLabel('', self)
        self.trial_label.setGeometry(50, 150, 100, 50)

        self.experiment_time_label = QLabel('Elapsed Time:', self)
        self.experiment_time_label.setGeometry(150, 150, 150, 50)

        self.time_label = QLabel('', self)
        self.time_label.setGeometry(300, 150, 100, 50)

        self.show()

    def check_time(self):
        # we have to use this instead
        elapsed_time_syn = self.syn.getSystemStatus()['recordSecs']
        # this is for showing only
        elapsed_time = self.start_time.elapsed()
        if elapsed_time_syn >= self.exp_duration_seconds:
            self.syn.setModeStr('Idle')
            self.timer.stop()
            self.close()
        else:
            self.elapsed_time = self.elapsed_time.addMSecs(self.timer.interval())
            time_str = self.elapsed_time.toString('hh:mm:ss')
            self.time_label.setText(time_str)
        

    def on_train_button_clicked(self):
        sender = self.sender()
        train_idx = self.train_buttons.index(sender)
        train_name = self.train_names[train_idx]

        trial_n = int(self.trial_label.text().split(' ')[1])

        # get the trial parameters from the DataFrame
        trial_params = self.trials_df.query(f'trial_n == {trial_n}').iloc[0]

        # define parameter map
        param_map = {
            'Period1': 'trial_period_ms',
            'Width1': 'trial_width_ms',
            'Period2': 'stim_period_ms',
            'Width2': 'stim_width_ms',
            'Period3': 'light_period_ms',
            'Width3': 'light_width_ms'
        }

        # set parameter values
        for param_name, trial_key in param_map.items():
            value = trial_params[trial_key]
            self.syn.setParameterValue(train_name, param_name, value)
            print(f"Trial {trial_n}: {train_name} Set {param_name} to {value} using {trial_key}")

        print(f'Trial {trial_n}: Ready to press {train_name}!')

        # update the trial label
        trial_n += 1
        self.trial_label.setText(f'Trial {trial_n}')


if __name__ == '__main__':
    # define experiment parameters
    train_names = ['Train1', 'Train2']
    train_labels = ['Train 1', 'Train 2']

    # initialize connection to experimental device
    syn = tdt.SynapseAPI()
    #syn = []
    syn.setModeStr('Preview')
    # connect to the device and set the mode to Active

    # create the GUI
    app = QApplication(sys.argv)

    # set Wayland environment variable on Ubuntu
    if QSysInfo.productType() == 'ubuntu':
        os.environ['QT_QPA_PLATFORM'] = 'wayland'

    # Generate the trial structure
    lights_df = generate_light_trials(frequencies_hz, weights, light_width_ms, n_trials)
    stim_df = generate_stim_trials(stim_duty_cycles, stim_weights, stim_period_ms, n_trials)
    lights_df = lights_df.merge(stim_df, on="trial_n", suffixes=('',''))

    # Add the Trial Period and Width
    lights_df['trial_period_ms'] = trial_period_ms
    lights_df['trial_width_ms'] = trial_width_ms

    gui = ExperimentGUI(syn, trials_df = lights_df, train_names = train_names, train_labels = train_labels)

    # run the event loop
    sys.exit(app.exec_())

    # Our desired elapsed time has passed, switch to Idle mode
    syn.setModeStr('Idle')
    print('done')
