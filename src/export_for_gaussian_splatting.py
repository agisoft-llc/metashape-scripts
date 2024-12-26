# export_for_gaussian_splatting.py is discontinued because corresponding functionality now included in Metashape.
# Please, use instead:
#   File -> Export -> Export Cameras...
# And choose 'Colmap (*.txt)' in 'Files of type'.

import Metashape
from PySide2 import QtWidgets

# Checking compatibility
compatible_major_version = "2.2"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

def export_for_gaussian_splatting_gui():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Error")
    layout = QtWidgets.QVBoxLayout(dialog)

    label = QtWidgets.QLabel(
        "export_for_gaussian_splatting.py is discontinued because corresponding functionality now included in Metashape 2.2.<br>"
        "Please, use instead:<br>"
        "<b>File->Export->Export Cameras...</b><br>"
        "And choose '<i>Colmap (*.txt)</i>' in 'Files of type'."
    )
    label.setWordWrap(False)
    layout.addWidget(label)

    button = QtWidgets.QPushButton("OK")
    button.clicked.connect(dialog.accept)
    layout.addWidget(button)

    dialog.exec()

label = "Scripts/[DISCONTINUED] Export Colmap project (for Gaussian Splatting)"
Metashape.app.addMenuItem(label, export_for_gaussian_splatting_gui)
print("To execute this script press {}".format(label))
