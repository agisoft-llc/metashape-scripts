import os
import sys
import shutil
import requests
from PySide2 import QtGui, QtCore, QtWidgets

"""
Script for managing Python scripts in Metashape.
Matjaž Mori, July 2024
https://github.com/matjash

The script provides a user interface for loading Python scripts into the Metashape environment. It includes the following features:
    Overview of Loaded Scripts: Displays a table with the names of loaded scripts and their file paths.
    Local Script Loading: Allows the user to load .py files from the local file system.
    Script Download: Fetches scripts from a GitHub repository and allows the user to select which ones
    to download and load.
    Script Execution: The script can be executed from the Metashape menu by selecting "Scripts/A Script Manager".
"""


GITHUB_REPO_API_URL = "https://api.github.com/repos/agisoft-llc/metashape-scripts/contents/src"

class LoadScriptDlg(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Script Manager")


        self.setMinimumSize(600, 400)


        # Help text
        current_script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        help_text = (f"<p>After loading new scripts, you need to restart Metashape to apply changes.</p>"
                     f"<p>Resources:</p>"
                     f"<ul>"
                     f"<li><a href='https://api.github.com/repos/agisoft-llc/metashape-scripts/contents/src'>Metashape Scripts Repository</a></li>"
                     f"<li><a href='file:///{current_script_dir}'>Local Metashape scripts folder</a></li>"
                     f"</ul>")
        
        self.helpLabel = QtWidgets.QLabel(help_text)
        self.helpLabel.setTextFormat(QtCore.Qt.RichText)
        self.helpLabel.setOpenExternalLinks(True)

        self.btnLoadLocal = QtWidgets.QPushButton("&Load Local Script")
        self.btnDownload = QtWidgets.QPushButton("&Download Scripts")
        self.btnQuit = QtWidgets.QPushButton("&Exit")

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Loaded Scripts", "File Path"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.open_menu)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setColumnHidden(1, True)

        layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()

        button_layout.addWidget(self.btnLoadLocal)
        button_layout.addWidget(self.btnDownload)
        button_layout.addWidget(self.btnQuit)

        layout.addWidget(self.helpLabel)
        layout.addWidget(self.table)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.btnLoadLocal.clicked.connect(self.load_local_scripts)
        self.btnDownload.clicked.connect(self.fetch_scripts)
        self.btnQuit.clicked.connect(self.reject)

        self.populate_table()
        self.exec()

    def populate_table(self):
        current_script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        self.table.setRowCount(0)
        
        for root, dirs, files in os.walk(current_script_dir):
            for file in files:
                if file.endswith('.py'):
                    row_position = self.table.rowCount()
                    self.table.insertRow(row_position)
                    self.table.setItem(row_position, 0, QtWidgets.QTableWidgetItem(file))
             
                    self.table.setItem(row_position, 1, QtWidgets.QTableWidgetItem(os.path.join(root, file)))
    
            


    def open_menu(self, position):
        menu = QtWidgets.QMenu()
        open_action = menu.addAction("Open")
        remove_action = menu.addAction("Remove")

        action = menu.exec_(self.table.mapToGlobal(position))
        if action == open_action:
            self.open_file()
        elif action == remove_action:
            self.remove_file()

    def open_file(self):
        selected_items = self.table.selectedItems()
        if selected_items:
            file_path = self.table.item(self.table.currentRow(), 1).text()
            if os.path.exists(file_path):
                os.startfile(file_path)

    def remove_file(self):
        selected_items = self.table.selectedItems()
        if selected_items:
            file_path = self.table.item(self.table.currentRow(), 1).text()
            if os.path.exists(file_path):
                os.remove(file_path)
                self.populate_table()

    def load_local_scripts(self):
        file_dialog = QtWidgets.QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select Scripts", "", "Python Files (*.py)")

        if file_paths:
            current_script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
            failed_files = []

            for file_path in file_paths:
                try:
                    destination_path = os.path.join(current_script_dir, os.path.basename(file_path))
                    if os.path.exists(destination_path):
                        reply = QtWidgets.QMessageBox.question(self, "File Exists",
                                                               f"The file {os.path.basename(file_path)} already exists. Do you want to overwrite it?",
                                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                               QtWidgets.QMessageBox.No)
                        if reply == QtWidgets.QMessageBox.No:
                            continue

                    shutil.copy(file_path, destination_path)
                except Exception as e:
                    failed_files.append(file_path)
                    print(f"Failed to copy {file_path}: {str(e)}")

            if failed_files:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Failed to copy some files:\n" + "\n".join(failed_files))
            else:
                QtWidgets.QMessageBox.information(self, "Success", "All scripts copied successfully. Please restart Metashape to apply changes.")
                self.populate_table()
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "No files selected")

    def fetch_scripts(self):
        response = requests.get(GITHUB_REPO_API_URL)
        if response.status_code != 200:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to fetch scripts from GitHub.")
            return
        
        files = self.parse_files(response.json())
        if not files:
            QtWidgets.QMessageBox.warning(self, "Error", "No scripts found.")
            return
        
        self.show_file_selection(files)
    
    def parse_files(self, data, path=""):
        files = []
        for item in data:
            if item['type'] == 'file' and item['name'].endswith('.py'):
                files.append((path + item['name'], item['download_url']))
            elif item['type'] == 'dir':
                subdir_files = requests.get(item['url']).json()
                files.extend(self.parse_files(subdir_files, path + item['name'] + '/'))
        return files

    def show_file_selection(self, files):
        file_dialog = QtWidgets.QDialog(self)
        file_dialog.setWindowTitle("Select Scripts to Load")
        file_dialog.setMinimumSize(600, 400)

        layout = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel("<p>Scripts available on <a href='https://github.com/agisoft-llc/metashape-scripts'>GitHub:</a></p>")
        label.setTextFormat(QtCore.Qt.RichText)
        label.setOpenExternalLinks(True)
        
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for file in files:
            self.file_list.addItem(file[0])
        
        btnDownload = QtWidgets.QPushButton("&Download Selected")
        btnCancel = QtWidgets.QPushButton("&Cancel")

        btnDownload.clicked.connect(lambda: self.download_files(files))
        btnCancel.clicked.connect(file_dialog.reject)

        layout.addWidget(label)
        layout.addWidget(self.file_list)
        layout.addWidget(btnDownload)
        layout.addWidget(btnCancel)

        file_dialog.setLayout(layout)
        file_dialog.exec()

    def download_files(self, files):
        selected_items = self.file_list.selectedItems()
        selected_files = [item.text() for item in selected_items]
        
        current_script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        failed_files = []

        for file, url in files:
            if file in selected_files:
                try:
                    destination_path = os.path.join(current_script_dir, os.path.basename(file))
                    if os.path.exists(destination_path):
                        reply = QtWidgets.QMessageBox.question(self, "File Exists",
                                                            f"The file {os.path.basename(file)} already exists. Do you want to overwrite it?",
                                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                            QtWidgets.QMessageBox.No)
                        if reply == QtWidgets.QMessageBox.No:
                            continue

                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(destination_path, 'w') as f:
                            f.write(response.text)
                    else:
                        failed_files.append(file)
                except Exception as e:
                    failed_files.append(file)
                    print(f"Failed to download {file}: {str(e)}")

        if failed_files:
            QtWidgets.QMessageBox.warning(self, "Warning", f"Failed to download some files:\n" + "\n".join(failed_files))
        else:
            QtWidgets.QMessageBox.information(self, "Success", "All selected scripts downloaded successfully. Please restart Metashape to apply changes.")
            self.populate_table()

def load_scripts_dialog():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = LoadScriptDlg(parent)

label = "Scripts/A Script Manager"
Metashape.app.addMenuItem(label, load_scripts_dialog)
print(f"To execute this script press {label}")
