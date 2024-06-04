import Metashape
from PySide2 import QtGui, QtCore, QtWidgets

# Checking compatibility
if Metashape.version < Metashape.Version(2, 1) or Metashape.version >= Metashape.Version(2, 2):
    raise Exception("Incompatible Metashape version: {}".format(Metashape.version))

class UndistortPhotosDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.resize(300, self.height())

        self.setWindowTitle("Undistort Photos")

        self.checkPrincipalPoint = QtWidgets.QCheckBox("Center principal point")
        self.checkSquarePixels = QtWidgets.QCheckBox("Square pixels")
        self.labelTemplate = QtWidgets.QLabel("Filename template")
        self.editTemplate = QtWidgets.QLineEdit("{filename}.{fileext}")

        self.buttonBox = QtWidgets.QDialogButtonBox()
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel);
        self.buttonBox.setCenterButtons(True)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.checkPrincipalPoint, 0, 0, 1, 2)
        layout.addWidget(self.checkSquarePixels, 1, 0, 1, 2)
        layout.addWidget(self.labelTemplate, 2, 0)
        layout.addWidget(self.editTemplate, 2, 1)
        layout.addWidget(self.buttonBox, 3, 0, 1, 2)
        self.setLayout(layout)

        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), self, QtCore.SLOT("accept()"))
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), self, QtCore.SLOT("reject()"))

    def accept(self):
        self.center_principal_point = self.checkPrincipalPoint.isChecked()
        self.square_pixels = self.checkSquarePixels.isChecked()
        self.filename_template = self.editTemplate.text()
        super().accept()

def undistort_photos():
    main_window = QtWidgets.QApplication.instance().activeWindow()

    dlg = UndistortPhotosDialog(main_window)
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return

    center_principal_point = dlg.center_principal_point
    square_pixels = dlg.square_pixels
    filename_template = dlg.filename_template

    folder = QtWidgets.QFileDialog.getExistingDirectory(main_window, "Select Output Folder")
    if folder == "":
        return

    path = folder + "/" + filename_template

    chunk = Metashape.app.document.chunk

    attached_sensors = set()

    if Metashape.version >= Metashape.Version(2, 1, 2):
        for c in chunk.cameras:
            if c.point_cloud is None:
                continue
            attached_sensors.add(c.sensor)

    old_calibrations = {}

    for s in chunk.sensors:
        if s.film_camera:
            continue

        calib = s.calibration
        if calib.type != Metashape.Sensor.Frame:
            continue

        if s in attached_sensors:
            continue

        user_calib = Metashape.Calibration()
        user_calib.type = calib.type
        user_calib.width = calib.width
        user_calib.height = calib.height
        user_calib.f = calib.f
        if not center_principal_point:
            user_calib.cx = calib.cx
            user_calib.cy = calib.cy
        if not square_pixels:
            user_calib.b1 = calib.b1

        old_calibrations[s.key] = s.user_calib
        s.user_calib = user_calib

    if Metashape.version >= Metashape.Version(2, 1, 2):
        chunk.convertImages(path, use_initial_calibration = True)
    else:
        task = Metashape.Tasks.ConvertImages()
        task.use_initial_calibration = True
        task.path = path
        task.apply(chunk)

    for s in chunk.sensors:
        s.user_calib = old_calibrations[s.key]

label = "Scripts/Undistort Photos"
Metashape.app.addMenuItem(label, undistort_photos)
print("To execute this script press {}".format(label))
