# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
from PySide2 import QtGui, QtCore, QtWidgets

# Checking compatibility
compatible_major_version = "1.5"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

QUALITY = {"1":  Metashape.UltraQuality,
           "2":  Metashape.HighQuality,
           "4":  Metashape.MediumQuality,
           "8":  Metashape.LowQuality,
           "16": Metashape.LowestQuality}

FILTERING = {"3": Metashape.NoFiltering,
             "0": Metashape.MildFiltering,
             "1": Metashape.ModerateFiltering,
             "2": Metashape.AggressiveFiltering}

MESH = {"Arbitrary": Metashape.SurfaceType.Arbitrary,
        "Height Field": Metashape.SurfaceType.HeightField}

DENSE = {"Ultra": Metashape.UltraQuality,
         "High": Metashape.HighQuality,
         "Medium": Metashape.MediumQuality,
         "Low": Metashape.LowQuality,
         "Lowest": Metashape.LowestQuality}


def isIdent(matrix):
    """
    Check if the matrix is identity matrix
    """
    for i in range(matrix.size[0]):
        for j in range(matrix.size[1]):
            if i == j:
                if matrix[i, j] != 1.0:
                    return False
            elif matrix[i, j]:
                return False
    return True


class SplitDlg(QtWidgets.QDialog):

    def __init__(self, parent):

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Split in chunks")

        self.gridX = 2
        self.gridY = 2
        self.gridWidth = 198
        self.gridHeight = 198

        self.spinX = QtWidgets.QSpinBox()
        self.spinX.setMinimum(1)
        self.spinX.setValue(2)
        self.spinX.setMaximum(20)
        self.spinX.setFixedSize(75, 25)
        self.spinY = QtWidgets.QSpinBox()
        self.spinY.setMinimum(1)
        self.spinY.setValue(2)
        self.spinY.setMaximum(20)
        self.spinY.setFixedSize(75, 25)

        self.chkMesh = QtWidgets.QCheckBox("Build Mesh")
        self.chkMesh.setToolTip("Generates mesh for each cell in grid")

        self.meshBox = QtWidgets.QComboBox()
        for element in MESH.keys():
            self.meshBox.addItem(element)

        self.chkDense = QtWidgets.QCheckBox("Build Dense Cloud")
        self.chkDense.setWhatsThis("Builds dense cloud for each cell in grid")

        self.denseBox = QtWidgets.QComboBox()
        for element in DENSE.keys():
            self.denseBox.addItem(element)

        self.chkMerge = QtWidgets.QCheckBox("Merge Back")
        self.chkMerge.setToolTip("Merges back the processing products formed in the individual cells")

        self.chkSave = QtWidgets.QCheckBox("Autosave")
        self.chkSave.setToolTip("Autosaves the project after each operation")

        self.txtOvp = QtWidgets.QLabel()
        self.txtOvp.setText("Overlap (%):")

        self.edtOvp = QtWidgets.QLineEdit()
        self.edtOvp.setPlaceholderText("0")
        self.edtOvp.setFixedSize(50, 25)

        self.btnQuit = QtWidgets.QPushButton("Close")
        self.btnQuit.setFixedSize(90, 50)

        self.btnP1 = QtWidgets.QPushButton("Split")
        self.btnP1.setFixedSize(90, 50)

        self.grid = QtWidgets.QLabel(" ")
        self.grid.resize(self.gridWidth, self.gridHeight)
        tempPixmap = QtGui.QPixmap(self.gridWidth, self.gridHeight)
        tempImage = tempPixmap.toImage()

        for y in range(self.gridHeight):
            for x in range(self.gridWidth):

                if not (x and y) or (x == self.gridWidth - 1) or (y == self.gridHeight - 1):
                    tempImage.setPixel(x, y, QtGui.qRgb(0, 0, 0))
                elif (x == self.gridWidth / 2) or (y == self.gridHeight / 2):
                    tempImage.setPixel(x, y, QtGui.qRgb(0, 0, 0))

                else:
                    tempImage.setPixel(x, y, QtGui.qRgb(255, 255, 255))

        tempPixmap = tempPixmap.fromImage(tempImage)
        self.grid.setPixmap(tempPixmap)
        self.grid.show()

        layout = QtWidgets.QGridLayout()  # creating layout
        layout.addWidget(self.spinX, 1, 0)
        layout.addWidget(self.spinY, 1, 1, QtCore.Qt.AlignRight)

        layout.addWidget(self.chkDense, 0, 2)
        layout.addWidget(self.chkMesh, 0, 3)
        layout.addWidget(self.chkMerge, 0, 4)

        layout.addWidget(self.meshBox, 1, 3, QtCore.Qt.AlignTop)
        layout.addWidget(self.denseBox, 1, 2, QtCore.Qt.AlignTop)

        layout.addWidget(self.chkSave, 3, 2)
        layout.addWidget(self.btnP1, 3, 3)
        layout.addWidget(self.btnQuit, 3, 4)

        layout.addWidget(self.txtOvp, 0, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.edtOvp, 0, 1, QtCore.Qt.AlignLeft)

        layout.addWidget(self.grid, 2, 0, 2, 2)
        # layout.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(layout)

        proc_split = lambda: self.splitChunks()

        self.spinX.valueChanged.connect(self.updateGrid)
        self.spinY.valueChanged.connect(self.updateGrid)

        QtCore.QObject.connect(self.btnP1, QtCore.SIGNAL("clicked()"), proc_split)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()

    def updateGrid(self):
        """
        Draw new grid
        """

        self.gridX = self.spinX.value()
        self.gridY = self.spinY.value()

        tempPixmap = QtGui.QPixmap(self.gridWidth, self.gridHeight)
        tempImage = tempPixmap.toImage()
        tempImage.fill(QtGui.qRgb(240, 240, 240))

        for y in range(int(self.gridHeight / self.gridY) * self.gridY):
            for x in range(int(self.gridWidth / self.gridX) * self.gridX):
                if not (x and y) or (x == self.gridWidth - 1) or (y == self.gridHeight - 1):
                    tempImage.setPixel(x, y, QtGui.qRgb(0, 0, 0))
                elif y > int(self.gridHeight / self.gridY) * self.gridY:
                    tempImage.setPixel(x, y, QtGui.qRgb(240, 240, 240))
                elif x > int(self.gridWidth / self.gridX) * self.gridX:
                    tempImage.setPixel(x, y, QtGui.qRgb(240, 240, 240))
                else:
                    tempImage.setPixel(x, y, QtGui.qRgb(255, 255, 255))

        for y in range(0, int(self.gridHeight / self.gridY + 1) * self.gridY, int(self.gridHeight / self.gridY)):
            for x in range(int(self.gridWidth / self.gridX) * self.gridX):
                tempImage.setPixel(x, y, QtGui.qRgb(0, 0, 0))

        for x in range(0, int(self.gridWidth / self.gridX + 1) * self.gridX, int(self.gridWidth / self.gridX)):
            for y in range(int(self.gridHeight / self.gridY) * self.gridY):
                tempImage.setPixel(x, y, QtGui.qRgb(0, 0, 0))

        tempPixmap = tempPixmap.fromImage(tempImage)
        self.grid.setPixmap(tempPixmap)
        self.grid.show()

        return True

    def splitChunks(self):

        self.gridX = self.spinX.value()
        self.gridY = self.spinY.value()
        partsX = self.gridX
        partsY = self.gridY

        print("Script started...")

        buildMesh = self.chkMesh.isChecked()
        buildDense = self.chkDense.isChecked()
        mergeBack = self.chkMerge.isChecked()
        autosave = self.chkSave.isChecked()

        quality = DENSE[self.denseBox.currentText()]
        mesh_mode = MESH[self.meshBox.currentText()]

        doc = Metashape.app.document
        chunk = doc.chunk

        original_chunk = chunk
        temporary_chunks = []

        if not chunk.transform.translation:
            chunk.transform.matrix = chunk.transform.matrix
        elif not chunk.transform.translation.norm():
            chunk.transform.matrix = chunk.transform.matrix
        elif chunk.transform.scale == 1:
            chunk.transform.matrix = chunk.transform.matrix
        elif isIdent(chunk.transform.rotation):
            chunk.transform.matrix = chunk.transform.matrix

        original_region = chunk.region
        r_center = original_region.center
        r_rotate = original_region.rot
        r_size = original_region.size

        x_scale = r_size.x / partsX
        y_scale = r_size.y / partsY
        z_scale = r_size.z

        offset = r_center - r_rotate * r_size / 2.

        for j in range(1, partsY + 1):  # creating new chunks and adjusting bounding box
            for i in range(1, partsX + 1):
                if not buildDense:
                    new_chunk = chunk.copy(items=[Metashape.DataSource.DenseCloudData, Metashape.DataSource.DepthMapsData])
                else:
                    new_chunk = chunk.copy(items=[])
                new_chunk.label = "Chunk " + str(i) + "_" + str(j)
                if new_chunk.model:
                    new_chunk.model.clear()

                temporary_chunks.append(new_chunk)

                new_region = Metashape.Region()
                new_rot = r_rotate
                new_center = Metashape.Vector([(i - 0.5) * x_scale, (j - 0.5) * y_scale, 0.5 * z_scale])
                new_center = offset + new_rot * new_center
                new_size = Metashape.Vector([x_scale, y_scale, z_scale])

                if self.edtOvp.text().isdigit():
                    new_region.size = new_size * (1 + float(self.edtOvp.text()) / 100)
                else:
                    new_region.size = new_size

                new_region.center = new_center
                new_region.rot = new_rot

                new_chunk.region = new_region

                Metashape.app.update()

                if autosave:
                    doc.save()

                if buildDense:
                    if new_chunk.depth_maps:
                        reuse_depth = True
                        if new_chunk.depth_maps.meta['depth/depth_downscale']:
                            quality = QUALITY[new_chunk.depth_maps.meta['depth/depth_downscale']]
                        if new_chunk.depth_maps.meta['depth/depth_filter_mode']:
                            filtering = FILTERING[new_chunk.depth_maps.meta['depth/depth_filter_mode']]
                        try:
                            task = Metashape.Tasks.BuildDepthMaps()
                            task.downscale = int(quality)
                            task.filter_mode = filtering
                            task.reuse_depth = reuse_depth
                            task.network_distribute = True
                            task.apply(new_chunk)

                            task = Metashape.Tasks.BuildDenseCloud()
                            task.max_neighbors = 100
                            task.network_distribute = True
                            task.point_colors = True
                            task.apply(new_chunk)
                        except RuntimeError:
                            print("Can't build dense cloud for " + chunk.label)

                    else:
                        reuse_depth = False
                        try:
                            task = Metashape.Tasks.BuildDepthMaps()
                            task.downscale = int(quality)
                            task.filter_mode = Metashape.FilterMode.MildFiltering
                            task.reuse_depth = reuse_depth
                            task.network_distribute = True
                            task.apply(new_chunk)

                            task = Metashape.Tasks.BuildDenseCloud()
                            task.max_neighbors = 100
                            task.network_distribute = True
                            task.point_colors = True
                            task.apply(new_chunk)
                        except RuntimeError:
                            print("Can't build dense cloud for " + chunk.label)

                    if autosave:
                        doc.save()

                if buildMesh:
                    if new_chunk.dense_cloud:
                        try:
                            new_chunk.buildModel(surface=mesh_mode,
                                                 source=Metashape.DataSource.DenseCloudData,
                                                 interpolation=Metashape.Interpolation.EnabledInterpolation,
                                                 face_count=Metashape.FaceCount.HighFaceCount)
                        except RuntimeError:
                            print("Can't build mesh for " + chunk.label)
                    else:
                        try:
                            new_chunk.buildModel(surface=mesh_mode,
                                                 source=Metashape.DataSource.PointCloudData,
                                                 interpolation=Metashape.Interpolation.EnabledInterpolation,
                                                 face_count=Metashape.FaceCount.HighFaceCount)
                        except RuntimeError:
                            print("Can't build mesh for " + chunk.label)
                    if autosave:
                        doc.save()

                if not buildDense:
                    if new_chunk.dense_cloud:
                        new_chunk.dense_cloud.clear()
                if new_chunk.depth_maps:
                    new_chunk.depth_maps.clear()
                # new_chunk = None

        if mergeBack:
            for chunk in temporary_chunks:
                chunk.remove(chunk.cameras)
            original_chunk.model = None  # hiding the mesh of the original chunk, just for case
            doc.mergeChunks([original_chunk] + temporary_chunks,
                            merge_dense_clouds=True, merge_models=True, merge_markers=True)  # merging all smaller chunks into single one
            merged_chunk = doc.chunks[-1]
            merged_chunk.region = original_region

            doc.remove(temporary_chunks)  # removing smaller chunks.
            if autosave:
                doc.save()

        if autosave:
            doc.save()

        print("Script finished!")
        return True


def split_in_chunks():
    global doc
    doc = Metashape.app.document

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = SplitDlg(parent)


label = "Custom menu/Split in chunks"
Metashape.app.addMenuItem(label, split_in_chunks)
print("To execute this script press {}".format(label))
