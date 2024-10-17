# Renders spherical panorama or regular capture of model from the current view point.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import math
import Metashape
from PySide2 import QtGui, QtCore, QtWidgets

view_consistent_direction = False

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    print("Unsupported Metashape version: {} != {}. Script may not work properly".format(found_major_version, compatible_major_version))


def render_image(source_data, sensor_type, result_width_px, result_height_px, center, rotation, fov, result_path):
    """
    Creates spherical panorama in the "center" point with the camera basis(right, up, back)
    transformed by the 3x3 "rotation" matrix.
    """

    chunk = Metashape.app.document.chunk
    y_up_z_back = Metashape.Matrix.Scale([1.0, -1.0, -1.0])

    calibration = Metashape.Calibration()

    if sensor_type == Metashape.Sensor.Type.Frame:
        calibration.f = result_height_px / 2 / math.tan(math.radians(fov / 2))
    elif sensor_type == Metashape.Sensor.Type.Spherical or sensor_type == Metashape.Sensor.Type.Fisheye:
        result_width_px = 2 * result_height_px
        calibration.f = result_height_px / math.pi

    calibration.width = result_width_px
    calibration.height = result_height_px 
    calibration.type = sensor_type

    transform = chunk.transform.matrix.inv() * Metashape.Matrix.Translation(center) * Metashape.Matrix.Rotation(rotation) * y_up_z_back

    if (source_data == Metashape.DataSource.ModelData):
        chunk.model.renderImage(transform, calibration).save(result_path)
    elif (source_data == Metashape.DataSource.PointCloudData):
        chunk.point_cloud.renderImage(transform, calibration, point_size=4).save(result_path)

class RenderImageDlg(QtWidgets.QDialog):

    def __init__ (self, parent):
        QtWidgets.QDialog.__init__(self)

        self.setWindowTitle("Render image")

        self.btnQuit = QtWidgets.QPushButton("&Cancel")
        self.btnOk = QtWidgets.QPushButton("&Ok")

        self.sourceDataTxt = QtWidgets.QLabel()
        self.sourceDataTxt.setText("Source data:")

        self.sourceDataCmb = QtWidgets.QComboBox()
        self.sourceDataCmb.addItem("Model")
        self.sourceDataCmb.addItem("Point cloud")

        self.sensorTypeTxt = QtWidgets.QLabel()
        self.sensorTypeTxt.setText("Sensor type:")

        self.sensorTypeCmb = QtWidgets.QComboBox()
        self.sensorTypeCmb.addItem("Frame")
        self.sensorTypeCmb.addItem("Spherical")
        self.sensorTypeCmb.addItem("Fisheye")        

        self.widthTxt = QtWidgets.QLabel()
        self.widthTxt.setText("Width:")
        self.widthSpinBox = QtWidgets.QSpinBox()
        self.widthSpinBox.setMaximum(1000000)
        self.widthSpinBox.setValue(1920);

        self.heightTxt = QtWidgets.QLabel()
        self.heightTxt.setText("Height:")
        self.heightSpinBox = QtWidgets.QSpinBox()
        self.heightSpinBox.setMaximum(1000000)
        self.heightSpinBox.setValue(1080);

        # creating layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)
        layout.addWidget(self.sourceDataTxt, 0, 0)
        layout.addWidget(self.sourceDataCmb, 0, 1)
        layout.addWidget(self.sensorTypeTxt, 1, 0)
        layout.addWidget(self.sensorTypeCmb, 1, 1)

        layout.addWidget(self.widthTxt, 2, 0)
        layout.addWidget(self.widthSpinBox, 2, 1)
        layout.addWidget(self.heightTxt, 3, 0)
        layout.addWidget(self.heightSpinBox, 3, 1)

        layout.addWidget(self.btnOk, 4, 0)
        layout.addWidget(self.btnQuit, 4, 1)
        self.setLayout(layout)

        QtCore.QObject.connect(self.btnOk, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("accept()"))
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))    

        self.exec()

    def accept(self):
        self.render_panorama_from_current_point()
        super().accept()

    def render_panorama_from_current_point(self):

        chunk = Metashape.app.document.chunk
        if (chunk == None):
            raise Exception("Null chunk")

        source_txt = self.sourceDataCmb.currentText()
        sensor_type_txt = self.sensorTypeCmb.currentText()

        source_data = Metashape.DataSource.ModelData
        sensor_type = Metashape.Sensor.Type.Frame

        if source_txt == "Model":
            source_data = Metashape.DataSource.ModelData
        elif source_txt == "Point cloud":
            source_data = Metashape.DataSource.PointCloudData

        if sensor_type_txt == "Frame":
            sensor_type = Metashape.Sensor.Type.Frame
        elif sensor_type_txt == "Spherical":
            sensor_type = Metashape.Sensor.Type.Spherical
        elif sensor_type_txt == "Fisheye":
            sensor_type = Metashape.Sensor.Type.Fisheye

        result_width_px  = self.widthSpinBox.value()
        result_height_px = self.heightSpinBox.value()

        if (source_data == Metashape.DataSource.ModelData):
            model = chunk.model
            if (model == None):
                raise Exception("Null model")
            if (model.getActiveTexture() == None):
                raise Exception("Model has no texture")

        if (source_data == Metashape.DataSource.PointCloudData):
            if (chunk.point_cloud == None):
                raise Exception("Null point cloud")

        result_path = Metashape.app.getSaveFileName("Resulting render path", filter="All Images (*.tif *.jpg *.png);;TIFF (*.tif);;JPEG (*.jpg);;PNG (*.png);;All Formats(*.*)");
        if (result_path == ""):
            print("No result path. Aborting")
            return 0

        viewpoint = Metashape.app.model_view.viewpoint
        view_center = viewpoint.center

        # z up, x front, y left
        z_up_rotation = chunk.crs.localframe(view_center).rotation().t() * Metashape.Matrix([
            [0,  0, -1],
            [-1, 0, 0 ],
            [0,  1, 0 ]
        ])

        view_rotation = viewpoint.rot
        rotation = view_rotation

        if (sensor_type == Metashape.Sensor.Type.Spherical and not view_consistent_direction):
            rotation = z_up_rotation

        print("Started rendering...")

        render_image(source_data, sensor_type, result_width_px, result_height_px, view_center, rotation, viewpoint.fov, result_path)
        
        print("Script finished!")
        
        return 1

def render_image_gui():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = RenderImageDlg(parent)

label = "Scripts/Render image"
Metashape.app.addMenuItem(label, render_image_gui)
print("To execute this script press {}".format(label))
