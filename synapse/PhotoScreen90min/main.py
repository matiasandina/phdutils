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
from datetime import timedelta
import time
import argparse
## Experiment PARAMS #####

# Set the experiment
sExperiment = "PhotoScreen90min"
exp_duration_hours = 1.5
exp_duration_seconds = exp_duration_hours * 3600

# We have a pTrain with ParameterNames
# ['ParameterList', 'StimActive', 'Mute', 'Strobe', 'Width1', 'Delay1', 'Period2', 'Width2', 'Period3', 'Width3']
timer_width_min = 30
# This goes into Width1
trial_period_ms = timer_width_min * 60 * 1000
# This goes into Delay1
trial_delay_ms = trial_period_ms
# This goes into Period2
stim_period_ms = 5 * 1000
# This goes into Width2 <- we want to change this width to increase the duty cycle and have less time off stim
stim_width_ms = 1 * 1000 
# This goes into Period3 <- we want to change this to change frequency of stimulation
light_period_ms = 40
# This goes into Width3 <- we want to keep this one here fixed
light_width_ms = 10


def set_subject(syn, id):
    known = syn.getKnownSubjects()
    if id in known:
        syn.setCurrentSubject(id)
    else:
        print(f"Creating new subject: {id}")
        syn.createSubject(id, '', 'mouse')
        syn.setCurrentSubject(id)


def set_experiment(syn, id):
    known = syn.getKnownExperiments()
    if id in known:
        syn.setCurrentExperiment(id)
    else:
        print("Experiment name not found. Check Script Name and SynapseAPI")
        sys.exit()


class ExperimentGUI(QWidget):
    def __init__(self, syn):
        super().__init__()
        self.syn = syn
        self.exp_duration_seconds = exp_duration_seconds

        self.init_ui()
        # Start the stimulus with params
        self.syn.setParameterValue('pTrain1', "Width1", trial_period_ms)
        self.syn.setParameterValue('pTrain1', "Delay1", trial_period_ms)
        self.syn.setParameterValue('pTrain1', "Period2", stim_period_ms)
        self.syn.setParameterValue('pTrain1', "Width2", stim_width_ms)
        self.syn.setParameterValue('pTrain1', "Period3", light_period_ms)
        self.syn.setParameterValue('pTrain1', "Width3", light_width_ms)
        
        self.syn.setParameterValue('pTrain1', "Strobe", 1)

        # initialize timer and elapsed time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_time)
        self.start_time = QTime.currentTime()
        self.elapsed_time = QTime(0, 0, 0)
        # start timer to track elapsed time
        self.timer.start(1000)

        # set trial label
        self.trial_label.setText('Habituation')

    def init_ui(self):
        # set up GUI elements
        self.setWindowTitle('Experiment GUI')
        self.setGeometry(100, 100, 500, 200)

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
            self.time_label.setText(f"{time_str}/{str(timedelta(seconds = exp_duration_seconds))}")
            if elapsed_time_syn >= timer_width_min * 60:
                self.trial_label.setText('Stim')
            if elapsed_time_syn >= timer_width_min * 60 * 2:
                self.trial_label.setText("Post Stim")
        

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='This script creates an experiment for screening mice with Opto stim. Check out the parameters at ')

    # Add arguments for ephys_folder and config_folder
    parser.add_argument('--subject_id', required=True, help='ID to create subject')

    # Parse the command-line arguments
    args = parser.parse_args()

    # initialize connection to experimental device
    syn = tdt.SynapseAPI()
    # Set subject and experiment
    set_experiment(syn, sExperiment)
    set_subject(syn, args.subject_id)
    syn.setModeStr('Record')
    time.sleep(1)
    print("Api Connected and Active")
    # connect to the device and set the mode to Active


    # create the GUI
    app = QApplication(sys.argv)

    # set Wayland environment variable on Ubuntu
    if QSysInfo.productType() == 'ubuntu':
        os.environ['QT_QPA_PLATFORM'] = 'wayland'

    gui = ExperimentGUI(syn)

    # run the event loop
    sys.exit(app.exec_())

    # Our desired elapsed time has passed, switch to Idle mode
    syn.setModeStr('Idle')
    print('done')
