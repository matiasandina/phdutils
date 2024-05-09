from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication
from signal_visualizer import SignalVisualizer
from styles import load_style
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    QCoreApplication.setApplicationName("Signal Visualizer")
    load_style(app)
    visualizer = SignalVisualizer()
    visualizer.show()
    sys.exit(app.exec_())
