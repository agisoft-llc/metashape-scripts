from PySide2 import QtGui, QtCore, QtWidgets
import shutil
import os
import sys

"""
Script for loading scripts into Metashape
Matjaz Mori, July 2024

The script provides a user interface for loading Python scripts into the Metashape environment. It includes the following features:

    File Selection: Allows the user to select one or more .py files to load.
    Overwrite Protection: Checks if a script already exists in the destination directory and prompts the user for confirmation before overwriting.
    Help Information: Displays help information, including a reminder to restart Metashape and links to useful resources.
    Resource Links: Provides clickable links to an online script repository and the local folder where scripts should be saved.
"""



class LoadScriptDlg(QtWidgets.QDialog):

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        self.setWindowTitle("Load Scripts")
        
        # Help text
        current_script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        help_text = (f"<p>After loading scripts, you need to restart Metashape to apply changes.</p>"
                     f"<p>Useful resources:</p>"
                     f"<ul>"
                     f"<li><a href='https://github.com/matjash/metashape-scripts/tree/master/src'>Metashape Scripts Repository</a></li>"
                     f"<li><a href='file:///{current_script_dir}'>Local folder where scripts should be saved</a></li>"
                     f"</ul>")

        self.helpLabel = QtWidgets.QLabel(help_text)
        self.helpLabel.setTextFormat(QtCore.Qt.RichText)
        self.helpLabel.setOpenExternalLinks(True)

        self.btnLoad = QtWidgets.QPushButton("&Load Scripts")
        self.btnQuit = QtWidgets.QPushButton("&Exit")

        layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()
        
        button_layout.addWidget(self.btnLoad)
        button_layout.addWidget(self.btnQuit)
        
        layout.addWidget(self.helpLabel)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.btnLoad.clicked.connect(self.load_scripts)
        self.btnQuit.clicked.connect(self.reject)

        self.exec()

    def load_scripts(self):
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
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "No files selected")

def load_scripts_dialog():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = LoadScriptDlg(parent)

label = "Scripts/Load Scripts"
Metashape.app.addMenuItem(label, load_scripts_dialog)
print(f"To execute this script press {label}")
