from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import polars as pl
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import minmax_scale, robust_scale
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt
from lspopt import spectrogram_lspopt
import datetime



class SpectrogramPlotWidget(pg.PlotWidget):
    doubleClicked = pyqtSignal(float)  # Define the new signal

    def __init__(self, parent=None):
        super().__init__(parent)

    def mouseDoubleClickEvent(self, event):
        mouse_point = self.plotItem.vb.mapSceneToView(event.pos())
        x = mouse_point.x()  # This gives the x coordinate of the mouse click, which corresponds to the time axis
        self.doubleClicked.emit(x)  # Emit a signal to update the position

