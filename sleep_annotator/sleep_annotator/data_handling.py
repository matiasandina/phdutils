from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import polars as pl
from halo import Halo
import time

class LoadThread(QThread):
    notifyProgress = pyqtSignal(int)  # Changed to emit integers representing progress
    dataLoaded = pyqtSignal(object)  # New signal that emits the loaded data

    def __init__(self, filename):
        QThread.__init__(self)
        self.filename = filename

    def run(self):
        # This does not naturally support progress updates
        # consider polars.read_csv_batched
        data = pl.read_csv(self.filename)  
        # Simulate progress for demonstration
        for percent_complete in range(101):  # Simulate loading
            time.sleep(0.05)  # Simulate time delay
            self.notifyProgress.emit(percent_complete)  # Emit progress update
        self.dataLoaded.emit(data)  # Emit loaded data once complete
        self.notifyProgress.emit('Data loaded.')


