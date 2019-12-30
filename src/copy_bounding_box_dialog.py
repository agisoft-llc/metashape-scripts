# Copies bounding boxes from chunk to other chunks.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
from PySide2 import QtGui, QtCore, QtWidgets

# Checking compatibility
compatible_major_version = "1.6"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


class CopyBoundingBoxDlg(QtWidgets.QDialog):

    def __init__(self, parent):

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Copy bounding box")

        self.labelFrom = QtWidgets.QLabel("From")
        self.labelTo = QtWidgets.QLabel("To")

        self.fromChunk = QtWidgets.QComboBox()
        for chunk in Metashape.app.document.chunks:
            self.fromChunk.addItem(chunk.label)

        self.toChunks = QtWidgets.QListWidget()
        self.toChunks.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for chunk in Metashape.app.document.chunks:
            self.toChunks.addItem(chunk.label)

        self.btnOk = QtWidgets.QPushButton("Ok")
        self.btnOk.setFixedSize(90, 50)
        self.btnOk.setToolTip("Copy bounding box to all selected chunks")

        self.btnQuit = QtWidgets.QPushButton("Close")
        self.btnQuit.setFixedSize(90, 50)

        layout = QtWidgets.QGridLayout()  # creating layout
        layout.addWidget(self.labelFrom, 0, 0)
        layout.addWidget(self.fromChunk, 0, 1)

        layout.addWidget(self.labelTo, 0, 2)
        layout.addWidget(self.toChunks, 0, 3)

        layout.addWidget(self.btnOk, 1, 1)
        layout.addWidget(self.btnQuit, 1, 3)

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnOk, QtCore.SIGNAL("clicked()"), self.copyBoundingBox)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()


    def copyBoundingBox(self):
        print("Script started...")

        doc = Metashape.app.document

        fromChunk = doc.chunks[self.fromChunk.currentIndex()]

        toChunks = []
        for i in range(self.toChunks.count()):
            if self.toChunks.item(i).isSelected():
                toChunks.append(doc.chunks[i])

        print("Copying bounding box from chunk '" + fromChunk.label + "' to " + str(len(toChunks)) + " chunks...")

        T0 = fromChunk.transform.matrix

        region = fromChunk.region
        R0 = region.rot
        C0 = region.center
        s0 = region.size

        for chunk in toChunks:

            if chunk == fromChunk:
                continue

            T = chunk.transform.matrix.inv() * T0

            R = Metashape.Matrix([[T[0, 0], T[0, 1], T[0, 2]],
                                  [T[1, 0], T[1, 1], T[1, 2]],
                                  [T[2, 0], T[2, 1], T[2, 2]]])

            scale = R.row(0).norm()
            R = R * (1 / scale)
            
            new_region = Metashape.Region()
            new_region.rot = R * R0
            c = T.mulp(C0)
            new_region.center = c
            new_region.size = s0 * scale / 1.

            chunk.region = new_region

        print("Script finished!")
        self.reject()


def copy_bbox():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = CopyBoundingBoxDlg(parent)


label = "Custom menu/Copy bounding box"
Metashape.app.addMenuItem(label, copy_bbox)
print("To execute this script press {}".format(label))
