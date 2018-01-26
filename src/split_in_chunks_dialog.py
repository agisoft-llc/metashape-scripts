# This is python script for PhotoScan Pro. Scripts repository: https://github.com/agisoft-llc/photoscan-scripts

import PhotoScan
from PySide2 import QtGui, QtCore, QtWidgets

# Checking compatibility
compatible_major_version = "1.4"
found_major_version = ".".join(PhotoScan.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible PhotoScan version: {} != {}".format(found_major_version, compatible_major_version))

QUALITY = {"1":  PhotoScan.UltraQuality,
           "2":  PhotoScan.HighQuality,
           "4":  PhotoScan.MediumQuality,
           "8":  PhotoScan.LowQuality,
           "16": PhotoScan.LowestQuality}

FILTERING = {"3": PhotoScan.NoFiltering,
             "0": PhotoScan.MildFiltering,
             "1": PhotoScan.ModerateFiltering,
             "2": PhotoScan.AggressiveFiltering}


class SplitDlg(QtWidgets.QDialog):

    def __init__(self, parent):

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Split in chunks")

        self.gridX = 2
        self.gridY = 2
        self.gridWidth = 198
        self.gridHeight = 198

        self.spinX = QtWidgets.QSpinBox()
        self.spinX.setMinimum(2)
        self.spinX.setMaximum(20)
        self.spinX.setFixedSize(75, 25)
        self.spinY = QtWidgets.QSpinBox()
        self.spinY.setMinimum(2)
        self.spinY.setMaximum(20)
        self.spinY.setFixedSize(75, 25)

        self.chkMesh = QtWidgets.QCheckBox("Build Mesh")
        self.chkMesh.setFixedSize(100, 50)
        self.chkMesh.setToolTip("Generates mesh for each cell in grid")

        self.chkDense = QtWidgets.QCheckBox("Build Dense Cloud")
        self.chkDense.setFixedSize(120, 50)
        self.chkDense.setWhatsThis("Builds dense cloud for each cell in grid")

        self.chkMerge = QtWidgets.QCheckBox("Merge Back")
        self.chkMerge.setFixedSize(90, 50)
        self.chkMerge.setToolTip("Merges back the processing products formed in the individual cells")

        self.chkSave = QtWidgets.QCheckBox("Autosave")
        self.chkSave.setFixedSize(90, 50)
        self.chkSave.setToolTip("Autosaves the project after each operation")

        self.txtOvp = QtWidgets.QLabel()
        self.txtOvp.setText("Overlap (%):")
        self.txtOvp.setFixedSize(90, 25)

        self.edtOvp = QtWidgets.QLineEdit()
        self.edtOvp.setPlaceholderText("0")
        self.edtOvp.setFixedSize(100, 25)

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
        layout.addWidget(self.spinX, 0, 0)
        layout.addWidget(self.spinY, 0, 1)

        layout.addWidget(self.chkDense, 0, 2)
        layout.addWidget(self.chkMesh, 0, 3)
        layout.addWidget(self.chkMerge, 0, 4)

        layout.addWidget(self.btnP1, 3, 2)
        layout.addWidget(self.btnQuit, 3, 3)

        layout.addWidget(self.txtOvp, 1, 3)
        layout.addWidget(self.edtOvp, 1, 4)

        layout.addWidget(self.chkSave, 2, 4)

        layout.addWidget(self.grid, 1, 0, 2, 2)
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

        doc = PhotoScan.app.document
        chunk = doc.chunk

        if not chunk.transform.translation:
            chunk.transform.matrix = chunk.transform.matrix

        region = chunk.region
        r_center = region.center
        r_rotate = region.rot
        r_size = region.size

        x_scale = r_size.x / partsX
        y_scale = r_size.y / partsY
        z_scale = r_size.z

        offset = r_center - r_rotate * r_size / 2.

        for j in range(1, partsY + 1):  # creating new chunks and adjusting bounding box
            for i in range(1, partsX + 1):
                new_chunk = chunk.copy(items=[PhotoScan.DataSource.DenseCloudData])
                new_chunk.label = "Chunk " + str(i) + "\\" + str(j)
                new_chunk.model = None

                new_region = PhotoScan.Region()
                new_rot = r_rotate
                new_center = PhotoScan.Vector([(i - 0.5) * x_scale, (j - 0.5) * y_scale, 0.5 * z_scale])
                new_center = offset + new_rot * new_center
                new_size = PhotoScan.Vector([x_scale, y_scale, z_scale])

                if self.edtOvp.text().isdigit():
                    new_region.size = new_size * (1 + float(self.edtOvp.text()) / 100)
                else:
                    new_region.size = new_size

                new_region.center = new_center
                new_region.rot = new_rot

                new_chunk.region = new_region

                PhotoScan.app.update()

                if autosave:
                    doc.save()

                if buildDense:
                    if new_chunk.depth_maps:
                        reuse_depth = True
                        quality = QUALITY[new_chunk.depth_maps.meta['depth/depth_downscale']]
                        filtering = FILTERING[new_chunk.depth_maps.meta['depth/depth_filter_mode']]
                        try:
                            new_chunk.buildDepthMaps(quality=quality, filter=filtering, reuse_depth=reuse_depth)
                            new_chunk.buildDenseCloud() #keep_depth=False 
                        except RuntimeError:
                            print("Can't build dense cloud for " + chunk.label)

                    else:
                        reuse_depth = False
                        try:
                            new_chunk.buildDepthMaps(quality=PhotoScan.Quality.HighQuality,
                                                     filter=PhotoScan.FilterMode.AggressiveFiltering, reuse_depth=reuse_depth)
                            new_chunk.buildDenseCloud() #keep_depth=False
                        except RuntimeError:
                            print("Can't build dense cloud for " + chunk.label)

                    if autosave:
                        doc.save()

                if buildMesh:
                    if new_chunk.dense_cloud:
                        try:
                            new_chunk.buildModel(surface=PhotoScan.SurfaceType.HeightField,
                                                 source=PhotoScan.DataSource.DenseCloudData,
                                                 interpolation=PhotoScan.Interpolation.EnabledInterpolation,
                                                 face_count=PhotoScan.FaceCount.HighFaceCount)
                        except RuntimeError:
                            print("Can't build mesh for " + chunk.label)
                    else:
                        try:
                            new_chunk.buildModel(surface=PhotoScan.SurfaceType.HeightField,
                                                 source=PhotoScan.DataSource.PointCloudData,
                                                 interpolation=PhotoScan.Interpolation.EnabledInterpolation,
                                                 face_count=PhotoScan.FaceCount.HighFaceCount)
                        except RuntimeError:
                            print("Can't build mesh for " + chunk.label)
                    if autosave:
                        doc.save()

                if not buildDense:
                    new_chunk.dense_cloud = None

                new_chunk.depth_maps = None
                # new_chunk = None

        if mergeBack:
            for i in range(1, len(doc.chunks)):
                chunk = doc.chunks[i]
                chunk.remove(chunk.cameras)
            doc.chunks[0].model = None  # removing model from original chunk, just for case
            doc.mergeChunks(doc.chunks,
                            merge_dense_clouds=True, merge_models=True, merge_markers=True)  # merging all smaller chunks into single one

            doc.remove(doc.chunks[1:-1])  # removing smaller chunks.
            if autosave:
                doc.save()

        if autosave:
            doc.save()

        print("Script finished!")
        return True


def split_in_chunks():
    global doc
    doc = PhotoScan.app.document

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = SplitDlg(parent)


label = "Custom menu/Split in chunks"
PhotoScan.app.addMenuItem(label, split_in_chunks)
print("To execute this script press {}".format(label))
