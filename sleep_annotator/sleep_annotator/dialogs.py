from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import polars as pl
import os

class FileSelectionDialog(QDialog):
    def __init__(self, filenames, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Select EEG Data File")
        layout = QVBoxLayout(self)
        self.listWidget = QListWidget(self)
        self.listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        layout.addWidget(self.listWidget)

        self.setMinimumSize(600, 300)  # Adjust as necessary

        # Store full filenames to map back from selection
        self.full_filenames = filenames

        # Add the filenames to the list widget, using the shortened path
        for filename in filenames:
            shortened_filename = self.shorten_path(filename)
            self.listWidget.addItem(shortened_filename)

        self.selected_file = None

        self.button = QPushButton('OK', self)
        layout.addWidget(self.button)
        self.button.clicked.connect(self.on_button_clicked)

    def on_button_clicked(self):
        if self.listWidget.currentItem():  # Check if an item is selected
            index = self.listWidget.currentRow()
            self.selected_file = self.full_filenames[index]  # Retrieve full path
        self.close()

    def getOpenFileName(self):
        self.exec_()
        return self.selected_file

    def shorten_path(self, path):
        parts = path.split(os.sep)
        # Ensure there are enough parts to process
        if len(parts) > 4:
            # Concatenate first two folders, ellipsis, and the last two segments
            shortened = os.sep.join(parts[:3] + ['...'] + parts[-2:])
        else:
            # If not enough parts, just join them normally
            shortened = os.sep.join(parts)
        return shortened



class FileDialog(QDialog):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        self.tree = QTreeView(self)
        layout.addWidget(self.tree)

        self.model = QFileSystemModel()
        self.model.setRootPath('')  # set the root path to the root of the filesystem
        self.tree.setModel(self.model)

        self.tree.clicked.connect(self.on_tree_clicked)

        self.button = QPushButton('OK', self)
        layout.addWidget(self.button)
        self.button.clicked.connect(self.on_button_clicked)

        self.selected_file = None

    def on_tree_clicked(self, index):
        self.selected_file = self.model.filePath(index)

    def on_button_clicked(self):
        self.close()

    def getOpenFileName(self):
        self.exec_()
        return self.selected_file, None

class LabelMappingDialog(QDialog):
    def __init__(self, unique_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Mapping")  # Set window title
        self.unique_values = unique_values
        self.mapping = {}
        layout = QVBoxLayout()
        
        # Add a description label
        description_label = QLabel("Please map the sleep states to the corresponding values in your data:")
        layout.addWidget(description_label)
        for state, state_name in {'1': "NREM", '2': "REM", '3': "Wake"}.items():
            hbox = QHBoxLayout()
            label = QLabel(f"{state_name} (Key {state}):")
            combo = QComboBox()
            combo.setEditable(True) # this allows people to add things
            combo.addItems(self.unique_values)
            hbox.addWidget(label)
            hbox.addWidget(combo)
            layout.addLayout(hbox)
            self.mapping[state] = combo
        # Create button box with "OK" and "Cancel" buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.validate_mapping)  # Validate the mapping when "OK" is clicked
        button_box.accepted.connect(self.accept)  # Close the dialog with accept result when "OK" is clicked
        button_box.rejected.connect(self.reject)  # Close the dialog with reject result when "Cancel" is clicked
        layout.addWidget(button_box)
        self.setLayout(layout)

    def get_mapping(self):
        result = {}
        for state, combo in self.mapping.items():
            result[state] = combo.currentText()
        return result

    def validate_mapping(self):
        for state, combo in self.mapping.items():
            value = combo.currentText()
            if value not in self.unique_values:
                QMessageBox.warning(self, "Warning", f"The value {value} might not found in the original data. Please confirm it's correct by inspecting console output.")


class LoadDataPage(QWizardPage):
    def __init__(self, intro_text, parent=None):
        super(LoadDataPage, self).__init__(parent)
        self.setTitle("Load Data")
        layout = QVBoxLayout(self)
        
        self.introLabel = QLabel(intro_text)
        layout.addWidget(self.introLabel)

        self.filePathEdit = QLineEdit(self)
        self.filePathEdit.setPlaceholderText("Enter the path to your data file...")
        self.fileButton = QPushButton("Browse...", self)
        self.fileButton.clicked.connect(self.browseFile)
        layout.addWidget(self.filePathEdit)
        layout.addWidget(self.fileButton)
        self.disclaimer = QLabel()
        layout.addWidget(self.disclaimer)
        self.setCommitPage(True)

        # Register the field with a wildcard to ensure it's required
        self.registerField('filePath*', self.filePathEdit)

    def browseFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'CSV Files (*.csv *.gz)')
        if filename:
            self.filePathEdit.setText(filename)


class PreviewDataPage(QWizardPage):
    def __init__(self, parent=None):
        super(PreviewDataPage, self).__init__(parent)
        self.setTitle("Preview Data")
        layout = QVBoxLayout(self)
        self.file_label = QLabel("File path will be shown here.")
        layout.addWidget(self.file_label)
        
        self.tableWidget = QTableWidget()
        layout.addWidget(self.tableWidget)

    def initializePage(self):
        # Hide the Back button when this page is active
        self.wizard().button(QWizard.BackButton).setVisible(False)
        filePath = self.field('filePath')
        if filePath:
            self.file_label.setText(f"This file was selected: {filePath}")
            self.loadPreviewData(filePath)

    def loadPreviewData(self, filePath):
        try:
            data = pl.read_csv(filePath, has_header=True, n_rows=10)  # Just a preview, not the full data
            self.populateTable(data)
        except Exception as e:
            self.file_label.setText(f"Failed to load data: {str(e)}")

    def populateTable(self, data):
        data_frame = data.to_pandas()
        self.tableWidget.setColumnCount(len(data_frame.columns))
        self.tableWidget.setRowCount(len(data_frame.index))
        self.tableWidget.setHorizontalHeaderLabels(data_frame.columns)
        
        for row in range(data_frame.shape[0]):
            for col in range(data_frame.shape[1]):
                self.tableWidget.setItem(row, col, QTableWidgetItem(str(data_frame.iat[row, col])))
        
        self.tableWidget.resizeColumnsToContents()


class MappingSetupPage(QWizardPage):
    def __init__(self, parent=None):
        super(MappingSetupPage, self).__init__(parent)
        self.setTitle("Setup Column Mapping")
        layout = QVBoxLayout(self)

        self.columnSelectLabel = QLabel("Select columns that contain categorical data:")
        layout.addWidget(self.columnSelectLabel)
        
        self.columnListWidget = QListWidget()
        self.columnListWidget.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.columnListWidget)
        
        self.manualValueEdit = QLineEdit(self)
        self.manualValueEdit.setPlaceholderText("Enter a value to map...")
        layout.addWidget(self.manualValueEdit)
        
        self.addValueButton = QPushButton("Add Value")
        self.addValueButton.clicked.connect(self.addValueToTable)
        layout.addWidget(self.addValueButton)

        self.inferButton = QPushButton("Infer Unique Values")
        self.inferButton.clicked.connect(self.inferUniqueValues)
        layout.addWidget(self.inferButton)

        self.mappingTable = QTableWidget()
        self.mappingTable.setColumnCount(2)
        self.mappingTable.setHorizontalHeaderLabels(['Raw Value', 'Map To'])
        layout.addWidget(self.mappingTable)

        self.infoLabel = QLabel("Add possible categorical values and map them to desired categories.")
        layout.addWidget(self.infoLabel)

    def initializePage(self):
        filePath = self.field('filePath')
        if filePath:
            data = pl.read_csv(filePath, has_header=True)
            self.columnListWidget.clear()
            for col in data.columns:
                item = QListWidgetItem(col)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.columnListWidget.addItem(item)

    def inferUniqueValues(self):
        selected_items = [self.columnListWidget.item(i) for i in range(self.columnListWidget.count()) if self.columnListWidget.item(i).checkState() == Qt.Checked]
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select at least one column to infer from.")
            return

        filePath = self.field('filePath')
        data = pl.read_csv(filePath, has_header=True)
        unique_values = set()
        for item in selected_items:
            column_data = data.select(pl.col(item.text())).to_series().unique().to_list()
            unique_values.update(column_data)

        self.populateMappingTable(unique_values)
    
    def collectMappings(self):
        mappings = {}
        for row in range(self.mappingTable.rowCount()):
            original_value = self.mappingTable.item(row, 0).text()
            mapped_value = self.mappingTable.item(row, 1).text()
            if original_value and mapped_value:
                mappings[original_value] = mapped_value
        return mappings

    def validatePage(self):
        # This method will be called when the wizard is completed
        mappings = self.collectMappings()
        print("Sleep Data will be mapped using user-defined mappings:", mappings)
        return True

    def addValueToTable(self):
        raw_value = self.manualValueEdit.text()
        if raw_value:
            self.populateMappingTable([raw_value])
            self.manualValueEdit.clear()

    def populateMappingTable(self, values):
        for value in values:
            rowCount = self.mappingTable.rowCount()
            self.mappingTable.insertRow(rowCount)
            self.mappingTable.setItem(rowCount, 0, QTableWidgetItem(str(value)))
            self.mappingTable.setItem(rowCount, 1, QTableWidgetItem(""))  # For user to fill
        self.mappingTable.resizeColumnsToContents()

class DataWizard(QWizard):
    LoadPage, PreviewPage, MappingSetupPage = range(3)
    
    def __init__(self, intro_text, parent=None):
        super(DataWizard, self).__init__(parent)
        self.addPage(LoadDataPage(intro_text, self))
        self.addPage(PreviewDataPage(self))
        self.addPage(MappingSetupPage(self))
        self.setWindowTitle("Data Loading Wizard")
        self.mappings = {}  # This will store the final mappings

    def accept(self):
        # Called when the wizard is completed
        mapping_page = self.page(self.MappingSetupPage)
        if isinstance(mapping_page, MappingSetupPage):
            self.mappings = mapping_page.collectMappings()
        super().accept()