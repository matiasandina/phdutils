from halo import Halo
import sys
import time
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import polars as pl
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import minmax_scale, robust_scale
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt, decimate
from lspopt import spectrogram_lspopt
import datetime
from plotting import SpectrogramPlotWidget
from dialogs import DataWizard
from dialogs import FileSelectionDialog
from data_handling import LoadThread

class SignalVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_loaded = False
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.freq_label = QLabel("Sampling Frequency (Hz)", self)
        self.freq_input = QComboBox(self)
        # Populate the combo box with standard sampling frequencies
        standard_frequencies = ['64', '100', '128', '256', '512', '1000']
        self.freq_input.addItems(standard_frequencies)
        self.freq_input.setCurrentIndex(-1)  # Set to no selection initially
        self.freq_input.setToolTip("Set the sampling frequency for the data visualization in Hz.")
        self.freq_label.setToolTip("Set the sampling frequency for the data visualization in Hz.")
        self.freq_input.currentIndexChanged.connect(self.update_sampling_frequency)

        self.range_label = QLabel("Time Range (s)", self)
        self.range_label.setToolTip("Set the time in seconds for the data display window")
        self.range_input = QSpinBox(self)
        # When pressing enter, it should update plots
        self.range_input.editingFinished.connect(self.update_plots)
        self.range_input.setMinimum(1)
        self.range_input.setMaximum(10000)
        self.range_input.setValue(10) # 10 seconds min
        self.range_input.setToolTip("Set the time in seconds for the data display window")

        # widget for displaying the current time and entering a time to jump to
        self.time_input_label = QLabel("Current Time / Jump to Time", self)
        self.time_input = QLineEdit(self)
        self.time_input.setText("00:00:00")  # or whatever initial value is appropriate

        # button for triggering the jump to the entered time
        self.jump_button = QPushButton("Jump", self)
        self.jump_button.clicked.connect(self.jump_to_time)
        
        self.electrode_label = QLabel("Select Electrode for Spectrogram", self)
        self.electrode_input = QComboBox(self)
        self.electrode_selected = False
        self.electrode_input.currentTextChanged.connect(self.select_electrode)
        
        self.emg_label = QLabel("Select EMG", self)
        self.emg_input = QComboBox(self)
        self.emg_selected = False
        self.emg_input.currentTextChanged.connect(self.select_emg)

        # flags for computation
        self.previous_electrode = None
        self.need_recompute_spectrogram = True
        self.need_recompute_hilbert = True

        # Explanation and controls for setting EEG Y range
        self.eeg_y_range_controls = QWidget()
        self.eeg_y_range_layout = QVBoxLayout()
        self.eeg_y_range_explanation = QLabel("Set the Y range for the EEG plot.")
        self.eeg_y_range_explanation.setWordWrap(True)
        self.eeg_y_range_min = QSpinBox()
        self.eeg_y_range_min.setRange(-10000, 0)
        self.eeg_y_range_min.setValue(-600)
        self.eeg_y_range_min.valueChanged.connect(lambda: self.selected_electrode_plot.setYRange(self.eeg_y_range_min.value(), self.eeg_y_range_max.value()))
        self.eeg_y_range_max = QSpinBox()
        self.eeg_y_range_max.setRange(0, 10000)
        self.eeg_y_range_max.setValue(600)
        self.eeg_y_range_max.valueChanged.connect(lambda: self.selected_electrode_plot.setYRange(self.eeg_y_range_min.value(), self.eeg_y_range_max.value()))
        self.eeg_y_range_layout.addWidget(self.eeg_y_range_explanation)
        self.eeg_y_range_layout.addWidget(self.eeg_y_range_min)
        self.eeg_y_range_layout.addWidget(self.eeg_y_range_max)

        # Explanation and controls for setting EMG Y range
        self.emg_y_range_controls = QWidget()
        self.emg_y_range_layout = QVBoxLayout()
        self.emg_y_range_explanation = QLabel("Set the Y range for the EMG plot.")
        self.emg_y_range_explanation.setWordWrap(True)
        self.emg_y_range_min = QSpinBox()
        self.emg_y_range_min.setRange(-10000, 0)
        self.emg_y_range_min.setValue(-600)
        self.emg_y_range_min.valueChanged.connect(lambda: self.selected_emg_plot.setYRange(self.emg_y_range_min.value(), self.emg_y_range_max.value()))
        self.emg_y_range_max = QSpinBox()
        self.emg_y_range_max.setRange(0, 10000)
        self.emg_y_range_max.setValue(600)
        self.emg_y_range_max.valueChanged.connect(lambda: self.selected_emg_plot.setYRange(self.emg_y_range_min.value(), self.emg_y_range_max.value()))
        self.emg_y_range_layout.addWidget(self.emg_y_range_explanation)
        self.emg_y_range_layout.addWidget(self.emg_y_range_min)
        self.emg_y_range_layout.addWidget(self.emg_y_range_max)

        # Ethogram ComboBox
        self.ann_data_mappings = {} # we will store mappings here
        self.ethogram_labels = None
        self.etho_label_label = QLabel("Ethogram Labels", self)
        self.etho_label_input = QComboBox(self)
        self.etho_label_selected = False
        self.etho_label_input.currentTextChanged.connect(self.select_etho_label)

        # Annotations 
        self.win_sec = 4  #seconds
        self.win_sec_label = QLabel("Annotation Window (s)", self)
        self.win_sec_label.setToolTip("Set the time in seconds the behavior epoch. This also determines the Spectrogram Resolution")
        self.win_sec_input = QLineEdit()
        self.win_sec_input.setText(str(self.win_sec))  # Set the initial value to self.win_sec
        self.win_sec_input.setToolTip("Set the time in seconds the behavior epoch. This also determines the Spectrogram Resolution")
        self.win_sec_button = QPushButton("Update")
        self.win_sec_button.clicked.connect(self.update_win_sec)  # Update self.win_sec when button is clicked

        # add autoscroll checkbox
        self.autoscroll_checkbox = QCheckBox("Annotate Autoscroll", self)
        self.autoscroll_checkbox.setChecked(True)
        self.layout.addWidget(self.autoscroll_checkbox)

        # Create a 'reset zoom' button
        self.spec_zoom_button = QPushButton("Spectrogram Zoom-In")
        self.spec_zoom_button.clicked.connect(self.spectrogram_zoom_in)
        self.reset_spec_zoom_button = QPushButton("Reset Spectrogram Zoom")
        self.reset_spec_zoom_button.clicked.connect(self.reset_spectrogram_zoom)

        self.setWindowTitle("Sleep Visualizer")

        # Top Menu
        self.exitAction = QAction(QIcon(), "Quit", self)
        self.exitAction.triggered.connect(self.confirmQuit)
        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('File')
        self.load_eeg_data_action = QAction('Load EEG Data', self)
        self.load_eeg_data_action.triggered.connect(self.load_eeg_data)
        self.load_ann_data_action = QAction('Load Annotation Data', self)
        self.load_ann_data_action.triggered.connect(self.load_ann_data)
        self.save_ann_data_action = QAction('Save Annotation Data', self)
        self.save_ann_data_action.triggered.connect(self.save_ann_data)
        self.save_ann_data_action.setShortcut("Ctrl+S")

        self.fileMenu.addAction(self.load_eeg_data_action)
        self.fileMenu.addAction(self.load_ann_data_action)
        self.fileMenu.addAction(self.save_ann_data_action)
        self.fileMenu.addAction(self.exitAction)

        # Set a container for the left panel
        self.input_layout = QVBoxLayout()
        self.input_container = QWidget()
        #self.input_container.setLayout(self.input_layout)

        # Add a QLabel widget to display the loaded filename
        self.filename_label = QLabel("No File Loaded")
        self.filename_label.setToolTip("No File Loaded")

        # Apply a custom style sheet to make the QLabel distinct
        self.filename_label.setStyleSheet("""
            background-color: #FF5733; /* Choose a background color that stands out */
            color: #ffffff; /* Text color */
            padding: 5px; /* Add some padding for spacing */
            border: 2px solid #FF5733; /* Add a border */
            """)

        # General Controls
        self.input_layout.addWidget(self.filename_label) # filename first
        self.input_layout.addWidget(self.freq_label)
        self.input_layout.addWidget(self.freq_input)
        self.input_layout.addWidget(self.range_label)
        self.input_layout.addWidget(self.range_input)
        self.input_layout.addWidget(self.time_input_label)
        self.input_layout.addWidget(self.time_input)
        self.input_layout.addWidget(self.jump_button)

        # EEG Controls
        self.eeg_group = QGroupBox("EEG Controls")
        self.eeg_layout = QVBoxLayout()
        self.eeg_layout.addWidget(self.electrode_label)
        self.eeg_layout.addWidget(self.electrode_input)
        self.eeg_y_range_layout = QHBoxLayout()
        self.eeg_y_range_layout.addWidget(QLabel("Y Min:"))
        self.eeg_y_range_layout.addWidget(self.eeg_y_range_min)
        self.eeg_y_range_layout.addWidget(QLabel("Y Max:"))
        self.eeg_y_range_layout.addWidget(self.eeg_y_range_max)
        self.eeg_layout.addWidget(self.eeg_y_range_explanation)
        self.eeg_layout.addLayout(self.eeg_y_range_layout)
        self.eeg_group.setLayout(self.eeg_layout)
        self.input_layout.addWidget(self.eeg_group)

        # Add a checkbox for toggling data scaling
        self.scale_data_checkbox = QCheckBox("Scale Data", self)
        self.scale_data_checkbox.setToolTip("Toggle to scale the data for better visualization")
        self.scale_data_checkbox.stateChanged.connect(self.toggle_data_scaling)
        self.eeg_layout.addWidget(self.scale_data_checkbox)

        # EMG Controls
        self.emg_group = QGroupBox("EMG Controls")
        self.emg_layout = QVBoxLayout()
        self.emg_layout.addWidget(self.emg_label)
        self.emg_layout.addWidget(self.emg_input)
        self.emg_y_range_layout = QHBoxLayout()
        self.emg_y_range_layout.addWidget(QLabel("Y Min:"))
        self.emg_y_range_layout.addWidget(self.emg_y_range_min)
        self.emg_y_range_layout.addWidget(QLabel("Y Max:"))
        self.emg_y_range_layout.addWidget(self.emg_y_range_max)
        self.emg_layout.addWidget(self.emg_y_range_explanation)
        self.emg_layout.addLayout(self.emg_y_range_layout)
        self.emg_group.setLayout(self.emg_layout)
        self.input_layout.addWidget(self.emg_group)

        # Ethogram Controls
        self.etho_group = QGroupBox("Ethogram Controls")
        self.etho_layout = QVBoxLayout()
        self.etho_layout.addWidget(self.etho_label_label)
        self.etho_layout.addWidget(self.etho_label_input)
        self.etho_group.setLayout(self.etho_layout)
        self.input_layout.addWidget(self.etho_group)

        # Spectrogram Controls
        self.spec_group = QGroupBox("Spectrogram Controls")
        self.spec_layout = QVBoxLayout()
        self.spec_layout.addWidget(self.spec_zoom_button)
        self.spec_layout.addWidget(self.reset_spec_zoom_button)
        self.spec_layout.addWidget(self.win_sec_label)
        self.spec_layout.addWidget(self.win_sec_input)
        self.spec_layout.addWidget(self.win_sec_button)
        self.spec_layout.addWidget(self.autoscroll_checkbox)
        self.spec_group.setLayout(self.spec_layout)
        self.input_layout.addWidget(self.spec_group)

        # Separator
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        self.input_layout.addWidget(self.separator)

        # Everything in the container into the layout
        #self.layout.addWidget(self.input_container)
        self.scroll_area = QScrollArea()
        self.input_container = QWidget()
        self.input_container.setLayout(self.input_layout)
        self.scroll_area.setWidget(self.input_container)
        self.layout.addWidget(self.scroll_area)

        # after adding all your widgets, add a spacer at the end
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.input_layout.addSpacerItem(spacer)


        # Spectrogram parameters
        self.spectrogram = QDockWidget("Spectrogram", self)
        self.spectrogram.setToolTip("Double-click to jump to time. Scroll to zoom-in/out")
        self.addDockWidget(Qt.RightDockWidgetArea, self.spectrogram)
        self.spectrogram_plot = SpectrogramPlotWidget()#pg.PlotWidget()  # Create a PlotWidget for the Spectrogram
        self.spectrogram.setWidget(self.spectrogram_plot)  # Set the PlotWidget as the dock widget's widget
        self.spectrogram_plot.setMouseEnabled(x=True, y=False)
        self.spectrogram_plot.doubleClicked.connect(self.jump_to_time_on_plot)
        self.spec_img = None
        # Initialize the flag to False
        self.spectrogram_zoomed = False
        # Get the ViewBox of the spectrogram plot and connect the sigRangeChanged signal
        self.spectrogram_plot.plotItem.vb.sigRangeChanged.connect(self.spectrogram_range_changed)

        # EMG Tab
        self.emg_tab_widget = QTabWidget()

        self.selected_emg = QDockWidget("EMG", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.selected_emg)
        self.selected_emg_plot = pg.PlotWidget()  # Create a PlotWidget for the Selected EMG
        # RMS EMG
        self.emg_rms_plot = pg.PlotWidget()  # Create a PlotWidget for the RMS EMG

        # Add the Y range controls and tab widget to a vertical layout
        self.emg_layout = QVBoxLayout()
        self.emg_layout.addWidget(self.emg_tab_widget)
        self.emg_container = QWidget()
        self.emg_container.setLayout(self.emg_layout)

        self.selected_emg.setWidget(self.emg_container)

        # Add the tabs
        self.emg_tab_widget.addTab(self.emg_rms_plot, "RMS EMG")
        self.emg_tab_widget.addTab(self.selected_emg_plot, "Selected EMG")

        # Selected eeg channel
        # Create a tab widget
        self.tab_widget = QTabWidget()

        # Create the dock widgets and add them to the main window
        self.eeg = QDockWidget("EEG", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.eeg)
        self.eeg_plot = pg.PlotWidget()  # Create a PlotWidget for the EEG
        # Add the tab widget to the dock widget
        self.eeg.setWidget(self.tab_widget)


        # Create a new plot widget for the selected electrode
        self.selected_electrode_plot = pg.PlotWidget()

        # Add the EEG plot and the selected electrode plot to the tab widget
        self.tab_widget.addTab(self.eeg_plot, "All Electrodes")
        self.tab_widget.addTab(self.selected_electrode_plot, "Selected Electrode")

        # Ethogram
        self.state_dict = {'1': "NREM", '2':"REM", '3':"Wake"}
        self.state_dict_text_keys = {v: k for k, v in self.state_dict.items()}
        # Define your color palettes
        palette1 = ['#3F6F76FF', '#69B7CEFF', '#C65840FF', '#F4CE4BFF', '#62496FFF']
        palette2 = ['#F7DC05FF', '#3D98D3FF', '#EC0B88FF', '#5E35B1FF', '#F9791EFF', '#3DD378FF', '#C6C6C6FF', '#444444FF']

        # Map stages to colors
        self.color_dict = {stage: color for stage, color in zip(self.state_dict.values(), palette1)}

        self.current_position = 0

        self.ethogram = QDockWidget("Ethogram", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.ethogram)
        self.ethogram_plot = pg.PlotWidget()  # Create a PlotWidget for the Ethogram
        self.ethogram_plot.setYRange(0, len(self.state_dict.keys()) + 1, padding=0)
        self.ethogram_plot.setMouseEnabled(x=True, y=False)
        self.ethogram.setWidget(self.ethogram_plot)  # Set the PlotWidget as the dock widget's widget

        self.power_plots = QDockWidget("Power Plots", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.power_plots)
        self.power_plots_plot = pg.PlotWidget()  # Create a PlotWidget for the Power Plots
        self.power_plots.setWidget(self.power_plots_plot)  # Set the PlotWidget as the dock widget's widget

        # Variables
        self.sampling_frequency = None
        self.time_range = self.range_input.value()


        # Link the x-axes of all the plots
        self.ethogram_plot.setXLink(self.eeg_plot)
        self.power_plots_plot.setXLink(self.eeg_plot)
        self.emg_rms_plot.setXLink(self.spectrogram_plot)

        # Key bindings
        # move data to the right
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.right_shortcut.activated.connect(self.move_right)
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.left_shortcut.activated.connect(self.move_left)

        for key, value in self.state_dict.items():
            shortcut = QShortcut(QKeySequence(key), self)
            # The lambda value=value bit is a way to "freeze" the current value of value to be used inside the annotate method. This is necessary because otherwise, Python's late binding behavior would cause all shortcuts to use the last value of value.
            shortcut.activated.connect(lambda key=key: self.annotate(key))

        # Key binding for the space bar
        self.space_bar_n = 4 # number of periods to do space_bar_n * self.win_sec
        self.space_bar_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_bar_shortcut.activated.connect(self.jump_space_bar)

        # Updating/listeners
        self.range_input.valueChanged.connect(self.update_position)
        #self.freq_input.valueChanged.connect(self.update_position)

        #self.setGeometry(300, 300, 1200, 600
        self.setWindowIcon(QIcon('logo.png'))
        self.showMaximized()
        self.central_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        #self.central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_layout.addStretch(1)
        self.input_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # eeg panel policy
        self.eeg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # window policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Modals
        self.munge_dialog = QDialog(self)
        self.munge_dialog.setWindowTitle("Processing Data")
        munge_layout = QVBoxLayout(self.munge_dialog)
        munge_layout.addWidget(QLabel("Processing data, please wait..."))
        self.munge_dialog.setLayout(munge_layout)
        self.munge_dialog.setModal(True)

        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Ready to load data!')


    def update_sampling_frequency(self, index):
        if index == -1:
            # No selection made
            self.prompt_for_frequency_selection()
        else:
            # Update the sampling frequency based on selected item
            self.sampling_frequency = int(self.freq_input.currentText())
    
    def prompt_for_frequency_selection(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Sampling Frequency")
        dialog.setModal(True)  # Makes the dialog blocking
        
        layout = QVBoxLayout()
        
        label = QLabel("Please select the sampling frequency of your input (Hz):")
        layout.addWidget(label)
        
        # Adding frequency options as buttons for example
        frequencies = ['64', '100', '128', '512', '1000']
        for freq in frequencies:
            btn = QPushButton(freq, dialog)
            btn.clicked.connect(lambda ch, f=freq: self.set_frequency_and_close(dialog, f))
            layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()  # This will block until the dialog is closed

    def set_frequency_and_close(self, dialog, freq):
        self.freq_input.setCurrentText(freq)
        self.sampling_frequency = int(self.freq_input.currentText())
        dialog.accept()  # Close the dialog

    def prepare_data_loading(self, filename):
        self.download_dialog = QDialog(self)
        self.download_dialog.setWindowTitle("Loading Data")
        layout = QVBoxLayout(self.download_dialog)

        # Setup progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Accessing data, please wait..."))

        self.download_dialog.setLayout(layout)
        self.download_dialog.setModal(True)

        # Setup thread
        self.load_thread = LoadThread(filename)
        self.load_thread.notifyProgress.connect(self.update_progress)
        self.load_thread.dataLoaded.connect(self.set_data)
        self.load_thread.finished.connect(self.finalize_data_loading)
        self.load_thread.start()

        self.download_dialog.exec_()  # This will block until the dialog is closed

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def finalize_data_loading(self):
        self.download_dialog.accept()  # Close the dialog
        loaded_filename = os.path.basename(self.load_thread.filename)
        self.filename_label.setText(f"Loaded File: {loaded_filename}")
        self.filename_label.setToolTip(loaded_filename)
        self.status_bar.showMessage('Data loaded and ready. Select electrodes to begin visualizing.')

    def load_eeg_data(self):
        # Ensure a valid frequency is selected
        if self.freq_input.currentIndex() == -1:
            self.prompt_for_frequency_selection()

        # Get directory from user
        directory = QFileDialog.getExistingDirectory(self, 'Open folder')
        if directory:
            # List CSV and GZ files in the directory
            filenames = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.csv', '.gz'))]
            if filenames:
                dialog = FileSelectionDialog(filenames)
                filename = dialog.getOpenFileName()
                if filename:
                    self.prepare_data_loading(filename)  # Start the data loading process
                else:
                    QMessageBox.warning(self, "File Selection", "No file was selected.")
            else:
                QMessageBox.warning(self, "File Selection", "No suitable files found in the directory.")


    def load_ann_data_directly(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'CSV Files (*.csv *.gz)')
        if not filename:
            QMessageBox.information(self, "Information", "No file selected.")
            return

        try:
            self.ann_data = pl.read_csv(filename)
            self.ethogram_labels = self.ann_data['labels'].to_numpy().astype(int)
            self.verify_annotation_length()
            if self.check_selections():
                self.update_ethogram_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error loading annotation data", str(e))

    def load_ann_data_with_mapping(self):
        intro_text = "Select a CSV file containing annotation data. This data will be previewed on the next page."
        wizard = DataWizard(intro_text, self)
        if wizard.exec_() == QWizard.Accepted:
            self.ann_data_mappings = wizard.mappings
            print("Mappings retrieved in visualizer")
            print(f"{self.ann_data_mappings}")
            filename = wizard.field('filePath')
            if filename:
                print(f"Trying to read selected file, this might take a while...")
                print(f"Reading {filename}")
                try:
                    self.ann_data = pl.read_csv(filename)
                    ann_data_names = self.ann_data.columns
                    self.etho_label_input.clear()  # Clear previous items
                    self.etho_label_input.addItems(ann_data_names)
                    if "time_sec" in self.ann_data.columns:
                        time_sec = self.ann_data.select(pl.col("time_sec")).to_numpy()
                        time_diff = np.diff(time_sec)
                        if not np.allclose(time_diff, self.win_sec):
                            result = QMessageBox.warning(self, 
                                "Warning", 
                                f"The difference in 'time_sec' is not consistent with selected window. Do you want to change the annotation window to {np.median(time_diff)} seconds?", 
                                QMessageBox.Yes | QMessageBox.No)
                            if result == QMessageBox.Yes:
                                self.win_sec = np.median(time_diff)
                except Exception as e:
                    QMessageBox.critical(self, "Error loading annotation data", str(e))
        else:
            QMessageBox.information(self, "Information", "File selection was cancelled.")


    def load_ann_data(self):
        # Start by asking if the data was scored by this visualizer
        response = QMessageBox.question(self, "Load Data", "Was the data scored using this Signal Visualizer?",
                                        QMessageBox.Yes | QMessageBox.No)

        if response == QMessageBox.Yes:
            self.load_ann_data_directly()
        else:
            self.load_ann_data_with_mapping() 

    def save_ann_data(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save file', '', 'CSV Files (*.csv *.gz)')
            if not filename:  # Check if filename is empty or None
                QMessageBox.information(self, "Information", "No file selected.")
                return

            # Check if the directory exists
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                QMessageBox.critical(self, "Error saving annotation data", f"Directory {directory} does not exist")
                return

            # Check if self.ethogram_labels is None or empty
            if self.ethogram_labels is None or len(self.ethogram_labels) == 0:
                QMessageBox.information(self, "Information", "No annotation data to save.")
                return

            # convert your ethogram_labels to a DataFrame and then save it
            ann_data = pl.DataFrame({
                "time_sec": np.arange(0, len(self.ethogram_labels) * self.win_sec, self.win_sec),
                "labels" : self.ethogram_labels,
                "behavior" : list(map(self.state_dict.get, [str(i) for i in self.ethogram_labels]))})
            # TODO: Polars does not support gzip yet?!
            # see https://github.com/pola-rs/polars/issues/13346
            # ann_data.write_csv(filename)
            ann_data.to_pandas().to_csv(filename, index=False) 

        except Exception as e:
            QMessageBox.critical(self, "Error saving annotation data", str(e))

    def remove_artifacts(df, column, upper_quantile=0.995, lower_quantile=0.005):
        # Compute the upper and lower bounds for clipping
        upper_bound = df[column].quantile(upper_quantile)
        lower_bound = df[column].quantile(lower_quantile)

        # Clip the values beyond the upper and lower bounds
        df = df.with_column(
            pl.when(df[column] > upper_bound)
            .then(upper_bound)
            .when(df[column] < lower_bound)
            .then(lower_bound)
            .otherwise(df[column])
            .alias(column)
        )
        return df

    def compute_hilbert(self):
        # Compute the Hilbert transform for each band for the entire dataset
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            #"alpha": (8, 13),
            #"beta": (13, 30),
            #"gamma": (30, 100),
        }

        # DataFrame to store the amplitude envelope of each band
        self.envelopes = pl.DataFrame()

        for band, (low, high) in bands.items():
            self.status_bar.showMessage(f"Computing hilbert for band {(low, high)}")
            print(f"Computing hilbert for band {(low, high)}")
            # Apply bandpass filter
            sos = butter(10, [low, high], btype='band', fs=self.sampling_frequency, output='sos')
            filtered = sosfiltfilt(sos, self.selected_electrode)

            # Apply Hilbert transform to get the envelope (i.e., the amplitude) of the signal
            analytic_signal = hilbert(filtered)
            amplitude_envelope = np.abs(analytic_signal)

            # Store the envelope in the DataFrame
            self.envelopes = self.envelopes.with_columns(pl.Series(band, amplitude_envelope))
        self.status_bar.showMessage("Hilbert Computation Finished")

    def spectrogram_range_changed(self):
        # This method is called whenever the user zooms in or out of the spectrogram
        # Set the flag to True to indicate that the user has zoomed in manually
        self.spectrogram_zoomed = True

    def spectrogram_zoom_in(self):
        # Get current x-axis range
        self.spectrogram_zoomed = True
        xmin = self.plot_from_buffer / self.sampling_frequency
        xmax = self.plot_to / self.sampling_frequency
        # Compute the center of the current range
        center = (xmin + xmax) / 2
        # Set new xmin and xmax based on self.time_range, keeping the center of the range the same
        new_xmin = center - self.time_range / 2
        new_xmax = center + self.time_range / 2
        # Set new x-axis range
        self.spectrogram_plot.setXRange(new_xmin, new_xmax)
        # Update spectrogram x-axis ticks
        self.update_x_axis_ticks()

    def reset_spectrogram_zoom(self):
        # This method is called when the user clicks the 'reset zoom' button
        # Reset the zoom level to the full range
        self.spectrogram_zoomed = False
        self.spectrogram_plot.plotItem.autoRange()
        self.update_x_axis_ticks()

    def compute_spectrogram(self, fmin=0.5, fmax=25):
        """
        Compute the multitaper spectrogram of the data using the Least-Squares 
        Spectral Analysis (LSSA) method.

        Parameters
        ----------
        data : :py:class:`numpy.ndarray`
            Single-channel EEG data. Must be a 1D NumPy array.
        sf : float
            The sampling frequency of data.
        win_sec : int or float
            The length of the sliding window, in seconds, used for multitaper PSD
            calculation. Default is 30 seconds.
        fmin, fmax : int or float
            The lower and upper frequency of the spectrogram. Default 0.5 to 25 Hz.

        Returns
        -------
        f : ndarray
            Array of sample frequencies.
        t : ndarray
            Array of segment times.
        Sxx : ndarray
            Spectrogram of x. The dimensions of Sxx are ``(len(f), len(t))``.
        """
        # Check the inputs
        assert isinstance(self.selected_electrode, np.ndarray), "Data must be a 1D NumPy array."
        assert self.selected_electrode.ndim == 1, "Data must be a 1D (single-channel) NumPy array."

        # Calculate multi-taper spectrogram
        nperseg = int(self.win_sec * self.sampling_frequency)
        print(f"computing spectrogram with win_sec = {self.win_sec} and nperseg = {nperseg}")
        assert self.selected_electrode.size > 2 * nperseg, "Data length must be at least 2 * win_sec."
        f, t, Sxx = spectrogram_lspopt(self.selected_electrode, self.sampling_frequency, nperseg=nperseg, noverlap=0)
        Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

        # Select only relevant frequencies
        good_freqs = np.logical_and(f >= fmin, f <= fmax)
        Sxx = Sxx[good_freqs, :]
        f = f[good_freqs]

        # shift t so that it starts at zero
        t = t - self.win_sec / 2
        self.status_bar.showMessage("Spectrogram computation finished")
        print("Spectrogram computation finished")
        return f, t, Sxx
            
    def apply_mappings(self, data, mappings):
        for column, map_dict in mappings.items():
            if column in data.columns:
                data[column] = data[column].apply(lambda x: map_dict.get(str(x), x))  # Ensure string conversion if needed
        return data

    def apply_user_mappings(self, data, column):
        # Mapping dictionary from string labels to integers for plotting
        label_to_int = {'Wake': 1, 'NREM': 2, 'REM': 3}

        # Apply mappings to convert labels to state names, then to integers
        if column in data.columns and column in self.ann_data_mappings:
            # Convert string labels to state names using user-defined mappings
            data[column] = data[column].apply(lambda x: self.ann_data_mappings[column].get(str(x), x))
            # Convert state names to integers for plotting
            data[column] = data[column].apply(lambda x: label_to_int.get(x, 0))  # Default to 0 if not found
        return data

    def map_back_to_original(self, labels):
        # Apply the original user_mapping to the labels
        return np.vectorize(self.ann_data_mappings.get)(labels)

    def verify_annotation_length(self):
        expected_length = len(self.selected_electrode) // (self.sampling_frequency * self.win_sec)
        actual_length = len(self.ethogram_labels)
        if actual_length != expected_length:
            QMessageBox.warning(self, "Warning",
                                f"The length of the annotations ({actual_length}) does not match the expected length ({expected_length}) based on the existing data and block size ({self.win_sec} sec).")

    
    def select_etho_label(self):
        if not self.check_selections():
            print("Waiting For EEG/EMG data to be loaded")
            return

        selected_column = self.etho_label_input.currentText()
        if not selected_column:
            print("No column selected.")
            return

        # Prepare labels for plotting: convert raw values to state names using mappings
        try:
            # Extract column data as numpy array of strings 
            # if input is int, this makes it 0 -> '0'
            raw_labels = self.ann_data[selected_column].to_numpy().astype(str)
            # Map raw labels to state names (NREM, REM, Wake) using user-provided mappings
            state_labels = np.vectorize(self.ann_data_mappings.get)(raw_labels, 'Unknown')  # 'Unknown' for unmapped values
            # Prepare a reverse mapping from state names to integers for plotting
            # the dictionary will have strings as the values ('1')
            self.ethogram_labels = np.vectorize(self.state_dict_text_keys.get)(state_labels, 0)  # Default to 0 for unknown states
            # actually coerce to int ('1' -> 1)
            self.ethogram_labels = self.ethogram_labels.astype(int)
            print(f"Processed unique labels for column {selected_column}: {np.unique(self.ethogram_labels)}")
            self.verify_annotation_length()
            if self.check_selections():
                self.update_ethogram_plot()
        except Exception as e:
            print(f"Error processing ethogram labels: {str(e)}")

    def find_closest_frequency(self, input_frequency, target=100):
        # Compute all possible divisors/factors of the input frequency
        divisors = [i for i in range(1, input_frequency+1) if input_frequency % i == 0]
        
        # Find the divisor that is closest to the target frequency
        closest_frequency = min(divisors, key=lambda x: abs(x - target))
        
        return closest_frequency

    def decimate_input_data(self, data, original_frequency):
        target_frequency = self.find_closest_frequency(original_frequency)
        downsample_factor = original_frequency // target_frequency
        # input data will be timestamps x channels. We shift polars -> np and then transpose for decimate to work in the rows
        original_columns = data.columns
        print(f"Input data shape is {data.shape}")
        data = data.to_numpy().T
        print(f"Downsampling with factor {downsample_factor} from {original_frequency} Hz to {target_frequency} Hz")
        # Decimate data using the calculated downsample factor
        data_down = decimate(data, downsample_factor, ftype='fir')
        data_down = pl.DataFrame(data_down.T, schema=original_columns)
        print(f"Decimated data shape is {data_down.shape}")
        return data_down, target_frequency

    # notoriously hard to clip the axes with a rule of thumb
    def return_clipped_range(self, array, clip = 0.8):
        min_val = np.min(array)
        max_val = np.max(array)
        return (min_val, min_val + clip * (max_val - min_val))

    def add_emg_diff(self):
        self.data = self.data.with_columns((pl.col("EMG1") - pl.col('EMG2')).alias('emg_diff'))
    
    def window_rms(self, signal, window_size):
        # let's make sure window_size is an int here
        window_size = int(window_size)
        num_segments = int(len(signal) // window_size)
        rms_values = np.zeros(num_segments)
        for i in range(num_segments):
            segment = signal[i * window_size: (i + 1) * window_size]
            rms_values[i] = np.sqrt(np.mean(segment ** 2))
        return rms_values

    def update_normalization(self):
        scaling_method = "robust" if self.scale_data_checkbox.isChecked() else None
        if scaling_method:
            self.eeg_plot_data = self.normalize_data(method=scaling_method)
            if self.check_selections():
                self.update_selected_eeg()
        else:
            self.eeg_plot_data = self.data  # Directly reference the original data without changes
            if self.check_selections():
                self.update_selected_eeg()

    def toggle_data_scaling(self):
        if self.scale_data_checkbox.isChecked():
            print("Data scaling applied")
        else:
            print("Displaying original data")
        self.update_normalization()

    def normalize_data(self, method=None):
        # If no method is specified, return the data as-is
        if method is None:
            return self.data
        assert method in ['robust', 'minmax'], f"Error: Scaling method must be either 'robust' (default) or 'minmax', received {method}"
        if method=="robust":
            return self.data.select(pl.all().map_batches(lambda x: pl.Series(robust_scale(x))))
        if method=="minmax":
            return self.data.select(pl.all().map_batches(lambda x: pl.Series(minmax_scale(x))))
    
    def update_selected_eeg(self):
        self.selected_electrode = self.eeg_plot_data.select(pl.col(self.electrode_input.currentText())).to_numpy().squeeze()
        # demean
        self.selected_electrode = self.selected_electrode - np.mean(self.selected_electrode)
        if self.check_selections():
            self.restore_plotting_axis()

            if self.need_recompute_spectrogram:
                print(f"Needed to recompute the spectrogram. Triggering from update_selected_electrode")
                self.spectrogram = self.compute_spectrogram()
                self.spec_img = None
                self.need_recompute_spectrogram = False  # Reset the flag
                self.status_bar.showMessage("Spectrogram updated.")

            if self.need_recompute_hilbert:
                    print(f"Needed to recompute hilbert. Triggering from update_selected_electrode")
                    self.compute_hilbert()
                    self.need_recompute_hilbert = False  # Reset the flag
                    self.status_bar.showMessage("Hilbert transform updated.")            

            # update all plots
            self.update_plots()
            # get to the current point
            self.jump_to_time()
        else:
            print("Please select EMG channel to trigger plot visualization")    
    
    def update_selected_emg(self):
        #self.selected_electrode_y_range = self.return_clipped_range(self.selected_electrode)
        self.selected_emg_channel = self.eeg_plot_data.select(pl.col(self.emg_input.currentText())).to_numpy().squeeze()
        # demean
        self.selected_emg_cannel = self.selected_emg_channel - np.mean(self.selected_emg_channel)
        self.log_rms_emg = self.window_rms(signal = self.selected_emg_channel, window_size = self.win_sec * self.sampling_frequency)
        self.log_rms_emg = np.log10(self.log_rms_emg)
        if self.check_selections():
            self.update_selected_eeg()
        else:
            print("Please select EEG channel to trigger plot visualization")   
    
    def select_electrode(self):
        # Check if a valid index is selected
        if self.electrode_input.currentIndex() != -1:
            current_electrode = self.electrode_input.currentText()
            if current_electrode != self.previous_electrode:
                print(f"Electrode changed from {self.previous_electrode} to {current_electrode}. Recomputing necessary data.")
                self.status_bar.showMessage(f"Electrode changed: {current_electrode}. Recomputing necessary data.")
                self.previous_electrode = current_electrode
                # Set flags to recompute since electrode has changed
                self.need_recompute_spectrogram = True
                self.need_recompute_hilbert = True
            else:
                print(f"Selected {current_electrode} as EEG electrode.")
                self.status_bar.showMessage(f"Selected {current_electrode} as EEG electrode.")

            self.electrode_selected = True
            self.update_selected_eeg()

    def select_emg(self):
        # Check if a valid index is selected
        if self.emg_input.currentIndex() != -1:
            print(f"Selected {self.emg_input.currentText()} as EMG electrode")
            self.status_bar.showMessage(f"Selected {self.emg_input.currentText()} as EMG electrode.")
            self.emg_selected = True
            self.update_selected_emg()

    def check_selections(self):
        # Check if both selections are made before munging data
        if self.electrode_selected and self.emg_selected:
            return True
        else:
            return False

    def restore_plotting_axis(self):
        # Determine data properties
        self.plot_from = 0
        self.current_position = 0
        self.plot_to = self.range_input.value() * self.sampling_frequency 
        self.sample_axis = np.arange(0, self.data.shape[0], 1)

    def set_data(self, data):
        if self.sampling_frequency > 100:
            data, new_frequency = self.decimate_input_data(data, self.sampling_frequency)
            self.sampling_frequency = new_frequency  # Update the frequency to the new downsampled rate
            self.freq_input.setCurrentText(str(self.sampling_frequency)) # update the new frequency in the UI
        # Set the original data
        self.data = data
        # Remove Outliers by clipping
        #self.data = self.remove_artifacts(self.data)

        # add emg diff assumes EMG1 - EMG2 is possible given names in data
        # self.add_emg_diff()
        # This will create self.eeg_plot_data
        self.update_normalization()
        self.restore_plotting_axis()
        self.time_range = self.range_input.value()

        self.data_loaded = True
        # Determine the number of complete windows in the data
        self.num_windows = int(self.data.shape[0] // (self.sampling_frequency * self.win_sec))
        # Initialize the ethogram_labels
        if self.ethogram_labels is None:
            self.ethogram_labels = np.zeros(self.num_windows, dtype=int)
        # Get the column names from the dataframe as a list of strings
        electrode_names = self.data.columns
        # Clear any old data from the QComboBox
        self.electrode_input.clear()
        self.emg_input.clear()
        # Disconnect signal
        self.electrode_input.currentTextChanged.disconnect()
        self.emg_input.currentTextChanged.disconnect()
        # Update combo box as needed
        self.electrode_input.addItems(electrode_names)
        self.emg_input.addItems(electrode_names)
        self.electrode_input.setCurrentIndex(-1)
        self.emg_input.setCurrentIndex(-1)
        # Reconnect signal
        self.electrode_input.currentTextChanged.connect(self.select_electrode)
        self.emg_input.currentTextChanged.connect(self.select_emg)

        self.current_position = 0

    def munge_data(self):
        #TODO: ADD THE MUNGE DIALOG
        #self.munge_dialog.show()
        self.update_normalization()
        self.update_selected_eeg()
        self.update_selected_emg()
        self.restore_plotting_axis()
        # Compute spectrogram
        self.spectrogram = self.compute_spectrogram()
        # Mark spec_img as none, so that update_spec re-plots
        self.spec_img = None
        # Compute power plots
        self.compute_hilbert()
        self.update_plots()
        # get to the current point
        self.jump_to_time()
        # dialog is having issues
        #self.munge_dialog.close()
        #self.spectrogram_plot.setFocus()

    def load_files_from_directory(self, folder_path):
        # Get list of .csv.gz files in the folder
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv.gz')]

        # Ask the user to select a file from the list
        file_name, ok = QInputDialog.getItem(self, "Select file", "Choose a file to load:", file_names, 0, False)
        if ok and file_name:  # Check if the user canceled the dialog
            full_path = os.path.join(folder_path, file_name)
            self.loader = LoadThread(full_path)
            self.loader.notifyProgress.connect(self.update_status)
            self.loader.start()

    def update_status(self, message):
        self.status_bar.showMessage(message)

    def update_current_time_input(self):
         # update the time displayed in the time_input
         current_time_sec = self.plot_from / self.sampling_frequency
         current_time = str(timedelta(seconds=current_time_sec))
         self.time_input.setText(current_time)

    def find_closest_window(self, time_in_seconds):
        # POTENTIALLY USEFUL, NOT USED FOR NOW
        window_size = self.sampling_frequency * self.win_sec
        # Calculate the nearest multiple of window_size
        closest_start = round(time_in_seconds / window_size) * window_size
        return closest_start

    def time_to_seconds(self, time_str):
        try:
            # Split the string by colon and handle fractional seconds
            time_parts = time_str.split(":")
            hours, minutes = int(time_parts[0]), int(time_parts[1])
            # Split seconds and fractional seconds
            seconds_parts = time_parts[2].split(".")
            seconds = int(seconds_parts[0])
            fractional_seconds = float("0." + seconds_parts[1]) if len(seconds_parts) > 1 else 0.0
            # Return total seconds as a float to preserve fractional part
            return hours * 3600 + minutes * 60 + seconds + fractional_seconds
        except (ValueError, IndexError):
            # Return None if the input is invalid
            return None

    def jump_to_time_on_plot(self, x):
        # x should be in seconds but let's remove the milliseconds
        x_seconds = int(x)
        # Convert seconds to HH:MM:SS
        x_time = str(timedelta(seconds=x_seconds))
        # now use x_time as your input for the time jump
        self.time_input.setText(x_time)
        self.jump_to_time()

    def jump_to_time(self):
        time_str = self.time_input.text()
        time_in_seconds = self.time_to_seconds(time_str)
        if time_in_seconds is not None:
            self.plot_from = int(time_in_seconds * self.sampling_frequency)
            self.plot_to = self.plot_from + self.range_input.value() * self.sampling_frequency
            self.update_current_time_input()
            self.current_position = int(self.time_to_seconds(self.time_input.text()) // self.win_sec)
            if self.data_loaded:
                self.update_plots()
                # Pan the spectrogram
                x_min, x_max = self.spectrogram_plot.viewRange()[0]
                #print(f"pan from {x_min, x_max}")
                current_range = x_max - x_min  # The current width of the viewing window
                # Determine the new x_min based on the target time
                new_x_min = time_in_seconds - current_range / 2  # Center the view on the target time
                new_x_max = time_in_seconds + current_range / 2
                #print(f"pan to {new_x_min, new_x_max}")
                # Ensure the new range is set within the bounds of your data
                new_x_min = max(new_x_min, 0)  # Assuming the data starts at 0
                new_x_max = new_x_max  # TODO: add here the greatest possible max_x
                self.spectrogram_plot.setXRange(new_x_min, new_x_max, padding=0)

        else:
            QMessageBox.warning(self, "Invalid input", "Please input a valid time in HH:MM:SS format.")

    def update_position(self):
        self.current_position = int(self.time_to_seconds(self.time_input.text()) // self.win_sec)
        self.plot_from = int(self.time_to_seconds(self.time_input.text()) * self.sampling_frequency)
        self.plot_to = self.plot_from + self.range_input.value() * self.sampling_frequency
        self.update_current_time_input()
        if self.data_loaded:
            # Before updating the plots, recalculate start_pos and end_pos for the shaded region
            start_pos = self.current_position * self.sampling_frequency * self.win_sec
            end_pos = (self.current_position + 1) * self.sampling_frequency * self.win_sec
            # It seems like you need to update the shaded region here, just like in update_eeg_plot()
            self.add_vertical_line(self.current_position * self.sampling_frequency * self.win_sec)
            self.add_shaded_region(start_pos, end_pos)
            self.update_plots()

    # Define the jump_space_bar function to perform the space bar action
    def jump_space_bar(self):
        current_time_sec = self.plot_from / self.sampling_frequency
        # Calculate the time to jump forward by self.space_bar_n * self.win_sec
        jump_time_seconds = self.space_bar_n * self.win_sec 
        next_time_seconds = current_time_sec + jump_time_seconds
        next_time = str(timedelta(seconds=next_time_seconds))
        # now use x_time as your input for the time jump
        self.time_input.setText(next_time)
        self.jump_to_time()


    def move_right(self):
        self.plot_from += int(self.sampling_frequency * self.win_sec)
        self.plot_to += int(self.sampling_frequency * self.win_sec)
        self.current_position = int(self.plot_from // (self.sampling_frequency * self.win_sec))
        self.update_current_time_input()
        if self.data_loaded:
            self.update_plots()
            # Add panning for spectrogram_plot
            x_min, x_max = self.spectrogram_plot.viewRange()[0]
            new_x_min = x_min + self.win_sec
            new_x_max = x_max + self.win_sec
            self.spectrogram_plot.setXRange(new_x_min, new_x_max, padding=0)


    def move_left(self):
        # prevent user from moving to negative
        if self.plot_from - self.sampling_frequency * self.win_sec >= 0:
            self.plot_from -= int(self.sampling_frequency * self.win_sec)
            self.plot_to -= int(self.sampling_frequency * self.win_sec)
            self.current_position = int(self.plot_from // (self.sampling_frequency * self.win_sec))
            self.update_current_time_input()
            if self.data_loaded:
                self.update_plots()
                # Add panning for spectrogram_plot
                x_min, x_max = self.spectrogram_plot.viewRange()[0]
                new_x_min = x_min - self.win_sec
                new_x_max = x_max - self.win_sec
                self.spectrogram_plot.setXRange(new_x_min, new_x_max, padding=0)
        else:
            print("Can't move from start to finish backwards")

    def update_win_sec(self):
        if self.ethogram_labels is not None:
            QMessageBox.warning(self, "CAUTION!", "Labels will not be interpolated to the new annotation window.\nChanging the annotation window can corrupt the data!")
        try:
            self.win_sec = float(self.win_sec_input.text())#int(self.win_sec_input.text())
            # Update time_range to be at least equal to win_sec
            if self.win_sec > self.time_range:
                print(f"Time range was lesser than annotation window, adjusting Time Range to {self.win_sec} seconds")
                self.range_input.setValue(int(self.win_sec))

            # If no data is loaded yet, return from the function
            if not self.data_loaded: 
                return
            else:
                self.spec_img = None  # Reset spec_img to force replot with new spectrogram data
                # Recompute the spectrogram     
                self.need_recompute_spectrogram = True
                self.update_selected_emg()
                

        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Please input a valid number.")


    def update_plots(self):
        # Fetch the required parameters
        self.sampling_frequency = self.sampling_frequency
        self.time_range = self.range_input.value()
        # Compute buffer
        self.plot_from_buffer = max(int(round(self.plot_from - self.win_sec * self.sampling_frequency)), 0)
        # Compute x-axis for plot: use self.plot_from_buffer here
        self.plot_x_axis = self.sample_axis[self.plot_from_buffer:self.plot_to]
        # Conversion from sample domain to time domain
        self.plot_x_axis_time = np.array(self.plot_x_axis) / self.sampling_frequency
        # Create tick labels
        self.x_tick_labels = np.arange(0, self.plot_x_axis_time[-1], self.win_sec)
        self.x_tick_labels_str = [self.pretty_time_label(seconds = i) for i in self.x_tick_labels]
        # Create the tick positions for these labels in the sample domain
        #self.x_tick_positions = self.x_tick_labels * self.sampling_frequency
        self.x_tick_positions = [int(round(label * self.sampling_frequency)) for label in self.x_tick_labels]
        # Call the update function for each plot
        self.update_eeg_plot()
        self.update_selected_emg_plot()
        self.update_spec_plot()
        self.update_ethogram_plot()
        self.update_power_plots()

    def pretty_time_label(self, seconds):
        # this function ensures that we have a format HH:MM:SS.f with only one decimal place
        stamp = datetime.datetime(1,1,1) + timedelta(seconds = seconds)
        return stamp.strftime('%H:%M:%S.%f')[:-5]
    # downsampling for x axis ticks
    def downsample(self, array, factor= 1):
        factor = self.sampling_frequency * factor
        return [val for i, val in enumerate(array) if divmod(i, factor)[1] == 0]

    def downsample_spec(self, array, factor =1):
        factor = self.win_sec * factor
        return [val for i, val in enumerate(array) if divmod(i, factor)[1] == 0]
    
    def update_eeg_plot(self):
        self.eeg_plot.clear()  # Clear previous plots
        self.selected_electrode_plot.clear()

        # Then plot the data
        for i, col in enumerate(self.data.columns):
            y_values = self.eeg_plot_data[col].to_numpy()[self.plot_from_buffer:self.plot_to]
            self.eeg_plot.plotItem.plot(self.plot_x_axis, y_values + 4 * i, pen=(i, self.data.shape[1]), name = f"Channel index {i}")

        self.eeg_plot.plotItem.autoRange()  # Force update the plot
        self.eeg_plot.getAxis("bottom").setTicks([list(zip(self.x_tick_positions, self.x_tick_labels_str))])
        self.eeg_plot.setLabel('left', "Electrical signal", units='uV')

        # Selected electrode
        self.selected_electrode_plot.plot(self.plot_x_axis, self.selected_electrode[self.plot_from_buffer:self.plot_to], pen=pg.mkPen('y', width=2), clear=True)
        self.selected_electrode_plot.getAxis("bottom").setTicks([list(zip(self.x_tick_positions, self.x_tick_labels_str))])
        self.selected_electrode_plot.setLabel('left', "Electrical signal", units='uV')
        #self.selected_electrode_plot.setYRange(*self.selected_electrode_y_range)
        # hardcoding 
        #self.selected_electrode_plot.setYRange(-600, 600)

        # Add vertical line at the current_position last, so it's on top
        # Add shaded region after line
        start_pos = self.current_position * self.sampling_frequency * self.win_sec
        end_pos = (self.current_position + 1) * self.sampling_frequency * self.win_sec
        self.add_vertical_line(start_pos)
        self.add_shaded_region(start_pos, end_pos)

    def add_vertical_line(self, pos):
        # EEG plot
        if hasattr(self, 'vLine_eeg'):
            self.eeg_plot.removeItem(self.vLine_eeg)

        self.vLine_eeg = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.vLine_eeg.setPos(pos)
        self.eeg_plot.addItem(self.vLine_eeg)

        # Selected electrode plot
        if hasattr(self, 'vLine_selected_electrode'):
            self.selected_electrode_plot.removeItem(self.vLine_selected_electrode)

        self.vLine_selected_electrode = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.vLine_selected_electrode.setPos(pos)
        self.selected_electrode_plot.addItem(self.vLine_selected_electrode)

        # Selected EMG plot
        if hasattr(self, 'vLine_selected_emg'):
            self.selected_emg_plot.removeItem(self.vLine_selected_emg)

        self.vLine_selected_emg = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.vLine_selected_emg.setPos(pos)
        self.selected_emg_plot.addItem(self.vLine_selected_emg)


    def add_shaded_region(self, start_pos, end_pos):
        # EEG plot
        if hasattr(self, 'region_eeg'):
            self.eeg_plot.removeItem(self.region_eeg)

        self.region_eeg = pg.LinearRegionItem([start_pos, end_pos], movable=False)
        self.region_eeg.setBrush((255, 255, 255, 50))  # Set color to white with alpha=50
        self.eeg_plot.addItem(self.region_eeg)

        # Selected electrode plot
        if hasattr(self, 'region_selected_electrode'):
            self.selected_electrode_plot.removeItem(self.region_selected_electrode)

        self.region_selected_electrode = pg.LinearRegionItem([start_pos, end_pos], movable=False)
        self.region_selected_electrode.setBrush((255, 255, 255, 50))  # Set color to white with alpha=50
        self.selected_electrode_plot.addItem(self.region_selected_electrode)

        # Selected EMG plot
        if hasattr(self, 'region_selected_emg'):
            self.selected_emg_plot.removeItem(self.region_selected_emg)

        self.region_selected_emg = pg.LinearRegionItem([start_pos, end_pos], movable=False)
        self.region_selected_emg.setBrush((255, 255, 255, 50))  # Set color to white with alpha=50
        self.selected_emg_plot.addItem(self.region_selected_emg)

    def update_selected_emg_plot(self):
        self.selected_emg_plot.clear()  # Clear previous plots

        # Select the data for the selected EMG within the current range
        emg_data = self.selected_emg_channel[self.plot_from_buffer:self.plot_to]

        # Plot the EMG data
        self.selected_emg_plot.plotItem.plot(self.plot_x_axis, emg_data, pen=pg.mkPen(color=(255, 255, 255), width=2))  # Plot in white

        # Deal with axes
        #self.selected_emg_plot.plotItem.autoRange()  # Force update the plot <- this will update both x and y, not useful
        self.selected_emg_plot.getAxis("bottom").setTicks([list(zip(self.x_tick_positions, self.x_tick_labels_str))])
        self.selected_emg_plot.setLabel('left', "Electrical signal", units='uV')

        # Add vertical line at the current_position last, so it's on top
        self.add_vertical_line(self.current_position * self.sampling_frequency * self.win_sec)

        # Add shaded region after line
        start_pos = self.current_position * self.sampling_frequency * self.win_sec
        end_pos = (self.current_position + 1) * self.sampling_frequency * self.win_sec
        self.add_shaded_region(start_pos, end_pos)
    
    def update_x_axis_ticks(self):
        # Get current x-axis range
        xmin, xmax = self.spectrogram_plot.viewRange()[0]
        # Determine the range in seconds
        range_seconds = xmax - xmin

        # Set ticks format and downsampling factor based on the range
        _, x_ticks_seconds, _ = self.spectrogram
        if range_seconds > 3600:  # if the range is more than an hour, use hours format
            x_ticks_labels = [str(int(i/3600)) for i in x_ticks_seconds]  # convert seconds to hours
            downsample_factor = 3600 / self.sampling_frequency  # we need to divide by sampling frequency because it gets multiplied by it at self.downsample
        elif range_seconds > 1000:  # if the range is less than or equal to an hour, use HH:MM:SS format
            x_ticks_labels = [self.pretty_time_label(seconds = i) for i in x_ticks_seconds]
            downsample_factor = 100  / self.sampling_frequency # adjust this value as needed
        else:
            x_ticks_labels = [self.pretty_time_label(seconds = i) for i in x_ticks_seconds]
            downsample_factor = 5 / self.sampling_frequency


        # Downsample x_ticks
        x_ticks_seconds = self.downsample(x_ticks_seconds, downsample_factor)
        x_ticks_labels = self.downsample(x_ticks_labels, downsample_factor)
        # Set x-axis ticks
        self.spectrogram_plot.getAxis("bottom").setTicks([list(zip(x_ticks_seconds, x_ticks_labels))])
        self.emg_rms_plot.getAxis("bottom").setTicks([list(zip(x_ticks_seconds, x_ticks_labels))])


    def update_spec_plot(self):
        # Check if spectrogram has been computed
        if self.spectrogram is None:
            return

        f, t, Sxx = self.spectrogram
        # Assuming the RMS values are computed with the same window size as the spectrogram
        t_rms = np.linspace(0, (len(self.log_rms_emg) - 1) * self.win_sec, len(self.log_rms_emg))

        # Check if the ImageItem for the spectrogram has been added to the spec_plot
        if self.spec_img is None:
            # Create ImageItem
            self.spec_img = pg.ImageItem()
            self.spectrogram_plot.addItem(self.spec_img)

            # Set color map: You might want to use different color map
            # https://colorcet.holoviz.org/
            color_map = pg.colormap.get('CET-D1')
            self.spec_img.setLookupTable(color_map.getLookupTable())

            # Set correct axis orientation for the image
            self.spec_img.setOpts(axisOrder='row-major')
            # Update the image displayed
            self.spec_img.setImage(Sxx)

            # Calculate the 5th and 95th percentile values
            vmin, vmax = np.percentile(Sxx, [5, 95])

            # Set the levels for the color map
            self.spec_img.setLevels([vmin, vmax])

            # Set the x and y ranges of the image to match the spectrogram data
            self.spec_img.setRect(QRectF(t[0], f[0], t[-1] - t[0], f[-1] - f[0]))

            # Set axis labels
            self.spectrogram_plot.setLabel('left', "Frequency", units='Hz')
            # Only auto-range if the user has not zoomed in manually
            if not self.spectrogram_zoomed:
                self.spectrogram_plot.plotItem.autoRange()

            # plot the rms
            self.emg_rms_plot.clear() # clear previous
            self.emg_rms_plot.plotItem.plot(t_rms, self.log_rms_emg, pen=pg.mkPen(color=(171, 235, 221), width=1))


        # Draw a vertical line indicating the current range
        # First, remove the previous line if it exists
        if hasattr(self, 'vLine'):
            self.spectrogram_plot.removeItem(self.vLine)
        if hasattr(self, 'vLine_rms'):
            self.emg_rms_plot.removeItem(self.vLine_rms)
        # Now add a new line
        start_time = self.current_position * self.win_sec  # Use current_position and window length to calculate start time
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.vLine.setPos(start_time)
        self.vLine.setBounds([t[0], t[-1]])
        self.spectrogram_plot.addItem(self.vLine)
        self.vLine_rms = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.vLine_rms.setPos(start_time)
        self.vLine_rms.setBounds([t_rms[0], t_rms[-1]]) # Assuming t_rms is the time array for the RMS plot
        self.emg_rms_plot.addItem(self.vLine_rms)

        # Set spectrogram x-axis ticks to HH:MM:SS format
        #x_ticks_seconds = t
        #x_ticks_labels = [str(timedelta(seconds=i)) for i in x_ticks_seconds]
        # downsample x_ticks for spectrogram
        #self.spectrogram_plot.getAxis("bottom").setTicks([list(zip(x_ticks_seconds, x_ticks_labels))])
        self.update_x_axis_ticks()
        # y ticks 
        y_ticks_labels = [0, 5, 10, 20]
        self.spectrogram_plot.getAxis("left").setTicks([list(zip(y_ticks_labels, map(str, y_ticks_labels)))])


    def update_power_plots(self):
        self.power_plots_plot.clear()  # Clear previous plots
        self.power_plots_plot_legend = self.power_plots_plot.addLegend()

        for i, band in enumerate(self.envelopes.columns):
            # Select the amplitude envelope for the current band within the current range
            amplitude_envelope = self.envelopes[band].to_numpy()[self.plot_from_buffer:self.plot_to]

            # Plot the envelope
            self.power_plots_plot.plotItem.plot(self.plot_x_axis, amplitude_envelope, pen=(i, len(self.envelopes.columns)), name = f"{band} band")

        self.power_plots_plot.plotItem.autoRange()  # Force update the plot
        self.power_plots_plot.getAxis("bottom").setTicks([list(zip(self.x_tick_positions, self.x_tick_labels_str))])

    
    def update_ethogram_plot(self):
        self.ethogram_plot.clear()  # Clear previous plots
        # Compute buffer in window units
        plot_from_window_units_buffer = int(round(self.plot_from_buffer // (self.win_sec * self.sampling_frequency)))
        # Convert plot_to from sample units to window units
        plot_to_window_units = int(round(self.plot_to // (self.win_sec * self.sampling_frequency)))

        # Calculate the lengths and values of consecutive segments in the ethogram labels
        lengths, values = self.rle(self.ethogram_labels[plot_from_window_units_buffer:plot_to_window_units])
        # Initialize the starting x-position for the first segment in window units
        x_start = plot_from_window_units_buffer
        
        # Iterate over each segment
        for length, value in zip(lengths, values):
            # If the label is zero, skip this segment (as it hasn't been annotated)
            if value == 0:
                x_start += length
                continue
            # Retrieve the state name and color associated with the label
            state_name = self.state_dict[str(value)]
            color = self.color_dict[state_name]
            
            # Convert x_start from window units to sample units and calculate x_end in sample units
            x_start_sample_units = x_start * self.win_sec * self.sampling_frequency
            x_end_sample_units = (x_start + length) * self.win_sec * self.sampling_frequency
            
            # Plot a bar for the current state, with its left and right edges at the starting and ending x-positions,
            # and its top and bottom edges at the y-position corresponding to the label and half unit above and below it
            self.ethogram_plot.addItem(pg.BarGraphItem(x0=[x_start_sample_units], x1=[x_end_sample_units], y0=[value - 0.5], y1=[value + 0.5], brush=color))
            
            # Update the starting x-position for the next segment in window units
            x_start += length
        
        #self.ethogram_plot.plotItem.autoRange()  # Force update the plot
        # Create list of tuples for Y-axis ticks
        y_tick_labels = [(float(i), '{} ({})'.format(name, i)) for i, name in self.state_dict.items()]
        self.ethogram_plot.getAxis('left').setTicks([y_tick_labels])
        # Fix x axis
        self.ethogram_plot.getAxis("bottom").setTicks([list(zip(self.x_tick_positions, self.x_tick_labels_str))])



    def annotate(self, label):
        # Update annotation
        self.ethogram_labels[self.current_position] = label
        # If autoscroll is enabled, move the position to the right
        if self.autoscroll_checkbox.isChecked():
            self.move_right()
        self.update_plots()  # To refresh the ethogram plot

    def rle(self, data):
        # Calculate the lengths and values of consecutive segments in the data
        lengths = np.diff(np.where(np.concatenate(([data[0]], data[:-1] != data[1:], [True])))[0])
        values = data[np.where(np.concatenate(([True], data[:-1] != data[1:])))[0]]
        return lengths, values

    def confirmQuit(self):
        # Display a message box to ask for confirmation
        reply = QMessageBox.question(self, "Quit", "Are you sure you want to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # User confirmed, quit the application
            QApplication.quit()
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Quit", "Are you sure you want to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
