from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import polars as pl
from halo import Halo
import time

class LoadThread(QThread):
    notifyProgress = pyqtSignal(str)
    dataLoaded = pyqtSignal(object)  # New signal that emits the loaded data

    def __init__(self, filename):
        QThread.__init__(self)
        self.filename = filename

    def run(self):
        with Halo(text='Loading data...', spinner='dots'):
            data = pl.read_csv(self.filename)  # Local variable to hold the data
            time.sleep(0.1)  # allow some time for spinner to spin
        self.dataLoaded.emit(data)  # Emit the loaded data
        self.notifyProgress.emit('Data loaded.')

