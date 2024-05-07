from PyQt5.QtWidgets import QApplication
import sys

def load_style(app):
    # Load base style
    with open("styles/base_style.qss", "r") as file:
        app.setStyleSheet(file.read())

    # Append OS-specific style
    if sys.platform.startswith('win'):
        with open("styles/windows_style.qss", "r") as file:
            app.setStyleSheet(app.styleSheet() + file.read())
    elif sys.platform.startswith('darwin'):
        with open("styles/mac_style.qss", "r") as file:
            app.setStyleSheet(app.styleSheet() + file.read())
    elif sys.platform.startswith('linux'):
        with open("styles/linux_style.qss", "r") as file:
            app.setStyleSheet(app.styleSheet() + file.read())
