# Dialog to remove all selected assets from selected chunks.
# For example, it might be useful to delete all orthophotos from the entire project to save storage space
# (if you don't plan to edit the orthomosaic afterwards).
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

from PySide2 import QtGui, QtCore, QtWidgets
import Metashape

# Checking compatibility
compatible_major_version = "2.2"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


class RemoveAssetsDlg(QtWidgets.QDialog):

    def __init__ (self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        self.setWindowTitle("Remove Assets")
        self.btnQuit = QtWidgets.QPushButton("&Exit")
        self.btnP1 = QtWidgets.QPushButton("&Remove")

        self.chunkTxt = QtWidgets.QLabel()
        self.chunkTxt.setText("Apply to:")
        self.radioBtn_all = QtWidgets.QRadioButton("All chunks")
        self.radioBtn_sel = QtWidgets.QRadioButton("Selected chunks")
        self.radioBtn_cur = QtWidgets.QRadioButton("Active chunk")
        self.radioBtn_all.setChecked(True)
        self.radioBtn_sel.setChecked(False)
        self.radioBtn_cur.setChecked(False)

        self.typeTxt = QtWidgets.QLabel()
        self.typeTxt.setText("Asset type:")
        self.typeCmb = QtWidgets.QComboBox()
        ASSET_TYPES = ["Key Points", "Tie Points", "Depth Maps", "Point Clouds", "Models", "Tiled Models", "DEMs", "Orthophotos", "Orthomosaics", "Shapes"]
        for type in ASSET_TYPES:
            self.typeCmb.addItem(type)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.typeTxt, 0, 0)
        layout.addWidget(self.typeCmb, 0, 1)
        layout.addWidget(self.chunkTxt, 1, 0)
        layout.addWidget(self.radioBtn_all, 1, 1)
        layout.addWidget(self.radioBtn_sel, 1, 2)
        layout.addWidget(self.radioBtn_cur, 1, 3)
        layout.addWidget(self.btnP1, 2, 1)
        layout.addWidget(self.btnQuit, 2, 2)
        self.setLayout(layout)

        QtCore.QObject.connect(self.btnP1, QtCore.SIGNAL("clicked()"), self.remove_assets)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()

    def remove_assets(self):
        doc = Metashape.app.document
        if not len(doc.chunks):
            print("Empty project, script aborted")
            return

        chunks = [doc.chunk]
        if self.radioBtn_sel.isChecked():
            chunks = [chunk for chunk in doc.chunks if chunk.selected]
        elif self.radioBtn_all.isChecked():
            chunks = list(doc.chunks)
        elif self.radioBtn_cur.isChecked():
            chunks = [doc.chunk]
        print("Selected chunks:")
        print(chunks)

        asset_type = self.typeCmb.currentText()
        for chunk in chunks:
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


def remove_assets_from_project():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = RemoveAssetsDlg(parent)


label = "Scripts/Remove assets"
Metashape.app.addMenuItem(label, remove_assets_from_project)
print("To execute this script press {}".format(label))
