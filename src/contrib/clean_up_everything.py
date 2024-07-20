import os
import sys
from PySide2 import QtGui, QtCore, QtWidgets

"""
Script for cleaning up Metashape projects by removing chosen assets.
Matjaž Mori, July 2024
https://github.com/matjash

"""


class CleanUpDlg(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clean Up Everything")

        # Layout
        self.layout = QtWidgets.QVBoxLayout(self)

        self.help_label = QtWidgets.QLabel(
            "This script will remove selected assets from one or more selected Metashape project files.\n"
            "You can choose which assets to remove and whether to apply the changes to the current project, "
            "selected projects, or all projects in subfolders."
        )
        self.layout.addWidget(self.help_label)


        # Checkboxes
        self.checkboxes = {
            "Key Points": QtWidgets.QCheckBox("Key Points"),
            "Tie Points": QtWidgets.QCheckBox("Tie Points"),
            "Depth Maps": QtWidgets.QCheckBox("Depth Maps"),
            "Point Clouds": QtWidgets.QCheckBox("Point Clouds"),
            "Models": QtWidgets.QCheckBox("Models"),
            "DEMs": QtWidgets.QCheckBox("DEMs"),
            "Orthophotos": QtWidgets.QCheckBox("Orthophotos"),
            "Orthomosaics": QtWidgets.QCheckBox("Orthomosaics"),
            "Shapes": QtWidgets.QCheckBox("Shapes"),
        }

        # Set default selections
        self.checkboxes["Orthophotos"].setChecked(True)
        self.checkboxes["Orthomosaics"].setChecked(True)
        self.checkboxes["DEMs"].setChecked(True)
        self.checkboxes["Point Clouds"].setChecked(True)

        for checkbox in self.checkboxes.values():
            self.layout.addWidget(checkbox)

        # Buttons
        self.remove_button = QtWidgets.QPushButton("Remove from This Project")
        self.remove_button.clicked.connect(self.remove_from_project)
        self.layout.addWidget(self.remove_button)

        self.select_button = QtWidgets.QPushButton("Select Project")
        self.select_button.clicked.connect(self.select_project)
        self.layout.addWidget(self.select_button)

        self.subfolders_button = QtWidgets.QPushButton("All Projects in Subfolders")
        self.subfolders_button.clicked.connect(self.remove_from_subfolders)
        self.layout.addWidget(self.subfolders_button)

        self.exit_button = QtWidgets.QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        self.layout.addWidget(self.exit_button)

        self.setLayout(self.layout)

    def get_selected_assets(self):
        return [asset for asset, checkbox in self.checkboxes.items() if checkbox.isChecked()]

    def confirm_removal(self, assets, projects):
        # Join assets with commas for a single line
        assets_message = ', '.join(assets)
        
        # Join projects with new lines to ensure each project is on a separate line
        projects_message = '\n'.join(projects)
        
        message = f"Are you sure you want to remove the following assets:\n\n" \
                f"{assets_message}\n\n" \
                f"From the following projects:\n\n" \
                f"{projects_message}"
        reply = QtWidgets.QMessageBox.question(self, 'Confirm Removal of celected assets', message,
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        return reply == QtWidgets.QMessageBox.Yes

    def handle_assets(self, chunk, asset_type):
        if asset_type == "Key Points":
            if chunk.tie_points:
                chunk.tie_points.removeKeypoints()
        elif asset_type == "Tie Points":
            chunk.tie_points = None
        elif asset_type == "Depth Maps":
            chunk.remove(chunk.depth_maps_sets)
        elif asset_type == "Point Clouds":
            chunk.remove(chunk.point_clouds)
        elif asset_type == "Models":
            chunk.remove(chunk.models)
        elif asset_type == "Tiled Models":
            chunk.remove(chunk.tiled_models)
        elif asset_type == "DEMs":
            chunk.remove(chunk.elevations)
        elif asset_type == "Orthophotos":
            for ortho in chunk.orthomosaics:
                ortho.removeOrthophotos()
        elif asset_type == "Orthomosaics":
            chunk.remove(chunk.orthomosaics)
        elif asset_type == "Shapes":
            chunk.shapes = None
        else:
            print("Unknown asset type: " + asset_type)
            return
        print(asset_type + " removed from " + chunk.label)

    def remove_from_project(self):
        # Assuming `Metashape` is available and `chunk` is obtained correctly
        chunk = Metashape.app.document.chunk  # Update this as per actual method to get the chunk
        selected_assets = self.get_selected_assets()
        assets = [asset for asset in selected_assets]
        projects = [chunk.label]  # Only one project in this case
        if self.confirm_removal(assets, projects):
            for asset in selected_assets:
                self.handle_assets(chunk, asset)

    def select_project(self):
        file_dialog = QtWidgets.QFileDialog(self, "Select Metashape Project")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Metashape Projects (*.psz *.psx)")
        file_dialog.setViewMode(QtWidgets.QFileDialog.List)
        file_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            selected_assets = self.get_selected_assets()
            if selected_assets:
                if self.confirm_removal(selected_assets, file_paths):
                    for file_path in file_paths:
                        doc = Metashape.Document()  # Assuming Metashape's Document class
                        doc.open(file_path)
                        chunk = doc.chunk
                        for asset in selected_assets:
                            self.handle_assets(chunk, asset)
                        doc.save()
                       

    def remove_from_subfolders(self):
        folder_dialog = QtWidgets.QFileDialog(self, "Select Folder")
        folder_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        folder_dialog.setOptions(QtWidgets.QFileDialog.DontUseNativeDialog)
        folder_dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        
        if folder_dialog.exec_():
            folder_path = folder_dialog.selectedFiles()[0]
            selected_assets = self.get_selected_assets()
            file_paths = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.psz', '.psx')):
                        file_paths.append(os.path.join(root, file))
            if selected_assets and file_paths:
                if self.confirm_removal(selected_assets, file_paths):
                    for file_path in file_paths:
                        doc = Metashape.Document()
                        doc.open(file_path)
                        chunk = doc.chunk
                        for asset in selected_assets:
                            self.handle_assets(chunk, asset)
                        doc.save()
                     

def clean_up_dialog():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = CleanUpDlg(parent)
    dlg.exec_()

label = "Scripts/Clean Up Everything"
Metashape.app.addMenuItem(label, clean_up_dialog)
print(f"To execute this script press {label}")
