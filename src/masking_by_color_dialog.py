# Creates masks for cameras in the active chunk, based on user defined color and tolerance.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
import tempfile, pathlib
from PySide2 import QtGui, QtCore, QtWidgets

# Checking compatibility
compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


class MaskByColor(QtWidgets.QDialog):

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        self.color = QtGui.QColor(0, 0, 0)
        red, green, blue = self.color.red(), self.color.green(), self.color.blue()

        self.setWindowTitle("Masking by color:")

        self.btnQuit = QtWidgets.QPushButton("Quit")
        self.btnQuit.setFixedSize(100, 50)

        self.btnP1 = QtWidgets.QPushButton("Mask")
        self.btnP1.setFixedSize(100, 50)

        self.pBar = QtWidgets.QProgressBar()
        self.pBar.setTextVisible(False)
        # self.pBar.setFixedSize(130, 50)

        self.operTxt = QtWidgets.QLabel()
        self.operTxt.setText("Operation:")
        self.operTxt.setFixedSize(100, 25)

        self.operValues = [("Replacement", Metashape.MaskOperation.MaskOperationReplacement),
                           ("Union", Metashape.MaskOperation.MaskOperationUnion),
                           ("Intersection", Metashape.MaskOperation.MaskOperationIntersection),
                           ("Difference", Metashape.MaskOperation.MaskOperationDifference)]
        self.operComboBox = QtWidgets.QComboBox()
        for label, value in self.operValues:
            self.operComboBox.addItem(label)

        self.selTxt = QtWidgets.QLabel()
        self.selTxt.setText("Apply to:")
        self.selTxt.setFixedSize(100, 25)

        self.radioBtn_all = QtWidgets.QRadioButton("All cameras")
        self.radioBtn_sel = QtWidgets.QRadioButton("Selected cameras")
        self.radioBtn_all.setChecked(True)
        self.radioBtn_sel.setChecked(False)

        self.colTxt = QtWidgets.QLabel()
        self.colTxt.setText("Select color:")
        self.colTxt.setFixedSize(100, 25)

        strColor = "{:0>2d}{:0>2d}{:0>2d}".format(int(hex(red)[2:]), int(hex(green)[2:]), int(hex(blue)[2:]))
        self.btnCol = QtWidgets.QPushButton(strColor)
        self.btnCol.setFixedSize(80, 25)
        pix = QtGui.QPixmap(10, 10)
        pix.fill(self.color)
        icon = QtGui.QIcon()
        icon.addPixmap(pix)
        self.btnCol.setIcon(icon)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Button, self.color)
        self.btnCol.setPalette(palette)
        self.btnCol.setAutoFillBackground(True)

        self.txtTol = QtWidgets.QLabel()
        self.txtTol.setText("Tolerance:")
        self.txtTol.setFixedSize(100, 25)

        self.sldTol = QtWidgets.QSlider()
        self.sldTol.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.sldTol.setMinimum(0)
        self.sldTol.setMaximum(99)

        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.operTxt, 0, 0)
        layout.addWidget(self.operComboBox, 1, 0)
        # layout.addWidget(self.selTxt, 0, 1)
        layout.addWidget(self.radioBtn_all, 0, 1)
        layout.addWidget(self.radioBtn_sel, 1, 1)
        layout.addWidget(self.colTxt, 0, 2)
        layout.addWidget(self.btnCol, 1, 2)
        layout.addWidget(self.txtTol, 0, 3)
        layout.addWidget(self.sldTol, 1, 3)

        layout.addWidget(self.pBar, 2, 0, 1, 2)
        layout.addWidget(self.btnP1, 2, 2)
        layout.addWidget(self.btnQuit, 2, 3)
        self.setLayout(layout)

        proc_mask = lambda: self.maskColor()
        proc_color = lambda: self.changeColor()

        QtCore.QObject.connect(self.btnP1, QtCore.SIGNAL("clicked()"), proc_mask)
        QtCore.QObject.connect(self.btnCol, QtCore.SIGNAL("clicked()"), proc_color)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()

    def changeColor(self):

        color = QtWidgets.QColorDialog.getColor()

        self.color = color
        red, green, blue = color.red(), color.green(), color.blue()
        if red < 16:
            red = "0" + hex(red)[2:]
        else:
            red = hex(red)[2:]
        if green < 16:
            green = "0" + hex(green)[2:]
        else:
            green = hex(green)[2:]
        if blue < 16:
            blue = "0" + hex(blue)[2:]
        else:
            blue = hex(blue)[2:]

        strColor = red + green + blue
        self.btnCol.setText(strColor)

        pix = QtGui.QPixmap(10, 10)
        pix.fill(self.color)
        icon = QtGui.QIcon()
        icon.addPixmap(pix)
        self.btnCol.setIcon(icon)

        palette = self.btnCol.palette()
        palette.setColor(QtGui.QPalette.Button, self.color)
        self.btnCol.setPalette(palette)
        self.btnCol.setAutoFillBackground(True)

        return True

    def maskColor(self):
        print("Masking...")

        tolerance = 10
        tolerance = self.sldTol.value()

        self.sldTol.setDisabled(True)
        self.btnCol.setDisabled(True)
        self.btnP1.setDisabled(True)
        self.btnQuit.setDisabled(True)

        chunk = Metashape.app.document.chunk
        mask_list = list()
        if self.radioBtn_sel.isChecked():
            for camera in chunk.cameras:
                if camera.selected and camera.type == Metashape.Camera.Type.Regular:
                    mask_list.append(camera)
        elif self.radioBtn_all.isChecked():
            mask_list = [camera for camera in chunk.cameras if camera.type == Metashape.Camera.Type.Regular]

        if not len(mask_list):
            Metashape.app.messageBox("Nothing to mask!")
            return False

        color = self.color
        red, green, blue = color.red(), color.green(), color.blue()

        _, mask_operation = self.operValues[self.operComboBox.currentIndex()]

        processed = 0
        for camera in mask_list:
            with tempfile.TemporaryDirectory() as temp_dir:
                for frame_index, frame in enumerate(camera.frames):
                    mask_file = str(pathlib.Path(temp_dir) / "mask_{key}_{frame}.png".format(key=camera.key, frame=frame_index + 1))
                    print(frame)
                    app.processEvents()
                    mask = Metashape.utils.createDifferenceMask(frame.photo.image(), (red, green, blue), tolerance, False)
                    mask.save(mask_file)
                    processed += 1
                    self.pBar.setValue(int(processed / len(mask_list) / len(chunk.frames) * 100))
                    chunk.frames[frame_index].generateMasks(cameras=[camera.key], path=mask_file, masking_mode=Metashape.MaskingMode.MaskingModeFile, mask_operation=mask_operation)

        print("Masking finished. " + str(processed) + " images masked.")

        self.sldTol.setDisabled(False)
        self.btnCol.setDisabled(False)
        self.btnP1.setDisabled(False)
        self.btnQuit.setDisabled(False)

        return True


def mask_by_color():
    global doc
    doc = Metashape.app.document

    global app
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = MaskByColor(parent)


label = "Scripts/Masking by color"
Metashape.app.addMenuItem(label, mask_by_color)
print("To execute this script press {}".format(label))
