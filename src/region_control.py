# Created by Vladimir Denisov ( https://www.agisoft.com/forum/index.php?topic=13969.0 )
# version 0.1.5 (with small modifications)
#
# Simple script for chunk region control. It move and scale to relative current transform.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import sys
import Metashape
from PySide2 import QtGui, QtCore, QtWidgets
from PySide2.QtWidgets import *

# Checking compatibility
compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


class ChunkRegionControl(QDialog):

    def __init__(self, parent):
        QDialog.__init__(self, parent)
        self.setWindowTitle("Region Control")
        self.vbox = QVBoxLayout()

        self.pos_gbox = QGroupBox("Region position")

        self.pos_gbox_layout = QVBoxLayout()
        self.pos_hbox = QHBoxLayout()
        self.pos_stepn_hbox = QHBoxLayout()
        self.pos_stepp_hbox = QHBoxLayout()
        self.pos_gbox.setLayout(self.pos_gbox_layout)

        self.pos_label = QLabel()
        self.pos_x = QLineEdit()
        self.pos_y = QLineEdit()
        self.pos_z = QLineEdit()

        self.pos_step_px = QPushButton()
        self.pos_step_px.clicked.connect(lambda: self.moveRegion(1))
        self.pos_step_py = QPushButton()
        self.pos_step_py.clicked.connect(lambda: self.moveRegion(3))
        self.pos_step_pz = QPushButton()
        self.pos_step_pz.clicked.connect(lambda: self.moveRegion(5))

        self.pos_step_nx = QPushButton()
        self.pos_step_nx.clicked.connect(lambda: self.moveRegion(2))
        self.pos_step_ny = QPushButton()
        self.pos_step_ny.clicked.connect(lambda: self.moveRegion(4))
        self.pos_step_nz = QPushButton()
        self.pos_step_nz.clicked.connect(lambda: self.moveRegion(6))

        self.buttons_phbox = QHBoxLayout()
        self.get_pos_btn = QPushButton()
        self.get_pos_btn.clicked.connect(lambda: self.getRegionPosition())
        self.set_pos_btn = QPushButton()
        self.set_pos_btn.clicked.connect(lambda: self.setRegionPosition())

        self.pos_label.setText("Position:")
        self.get_pos_btn.setText("Get position")
        self.set_pos_btn.setText("Set position")

        self.pos_step_nx.setText("Step X-")
        self.pos_step_ny.setText("Step Y-")
        self.pos_step_nz.setText("Step Z-")
        self.pos_step_px.setText("Step X+")
        self.pos_step_py.setText("Step Y+")
        self.pos_step_pz.setText("Step Z+")

        self.pos_hbox.addWidget(self.pos_label)
        self.pos_hbox.addWidget(self.pos_x)
        self.pos_hbox.addWidget(self.pos_y)
        self.pos_hbox.addWidget(self.pos_z)

        self.pos_stepn_hbox.addWidget(self.pos_step_nx)
        self.pos_stepn_hbox.addWidget(self.pos_step_ny)
        self.pos_stepn_hbox.addWidget(self.pos_step_nz)

        self.pos_stepp_hbox.addWidget(self.pos_step_px)
        self.pos_stepp_hbox.addWidget(self.pos_step_py)
        self.pos_stepp_hbox.addWidget(self.pos_step_pz)

        self.buttons_phbox.addWidget(self.get_pos_btn)
        self.buttons_phbox.addWidget(self.set_pos_btn)
        self.pos_gbox_layout.addLayout(self.pos_hbox)
        self.pos_gbox_layout.addLayout(self.buttons_phbox)
        self.pos_gbox_layout.addLayout(self.pos_stepn_hbox)
        self.pos_gbox_layout.addLayout(self.pos_stepp_hbox)

        self.size_gbox = QGroupBox("Region size")

        self.size_gbox_layout = QVBoxLayout()
        self.size_hbox = QHBoxLayout()
        self.size_reduce_hbox = QHBoxLayout()
        self.size_enlarge_hbox = QHBoxLayout()
        self.size_gbox.setLayout(self.size_gbox_layout)

        self.sz_label = QLabel()
        self.sz_x = QLineEdit()
        self.sz_y = QLineEdit()
        self.sz_z = QLineEdit()
        self.buttons_hsbox = QHBoxLayout()
        self.get_size_btn = QPushButton()
        self.get_size_btn.clicked.connect(lambda: self.getRegionSize())
        self.set_size_btn = QPushButton()
        self.set_size_btn.clicked.connect(lambda: self.setRegionSize())

        self.size_reducex_btn = QPushButton()
        self.size_reducex_btn.clicked.connect(lambda: self.reduceRegion(0))
        self.size_reducey_btn = QPushButton()
        self.size_reducey_btn.clicked.connect(lambda: self.reduceRegion(1))
        self.size_reducez_btn = QPushButton()
        self.size_reducez_btn.clicked.connect(lambda: self.reduceRegion(2))

        self.size_enlargex_btn = QPushButton()
        self.size_enlargex_btn.clicked.connect(lambda: self.enlargeRegion(0))
        self.size_enlargey_btn = QPushButton()
        self.size_enlargey_btn.clicked.connect(lambda: self.enlargeRegion(1))
        self.size_enlargez_btn = QPushButton()
        self.size_enlargez_btn.clicked.connect(lambda: self.enlargeRegion(2))

        self.sz_label.setText("Size:")
        self.get_size_btn.setText("Get size")
        self.set_size_btn.setText("Set size")

        self.size_reducex_btn.setText("Half X")
        self.size_reducey_btn.setText("Half Y")
        self.size_reducez_btn.setText("Half Z")

        self.size_enlargex_btn.setText("Double X")
        self.size_enlargey_btn.setText("Double Y")
        self.size_enlargez_btn.setText("Double Z")

        self.size_hbox.addWidget(self.sz_label)
        self.size_hbox.addWidget(self.sz_x)
        self.size_hbox.addWidget(self.sz_y)
        self.size_hbox.addWidget(self.sz_z)

        self.size_reduce_hbox.addWidget(self.size_reducex_btn)
        self.size_reduce_hbox.addWidget(self.size_reducey_btn)
        self.size_reduce_hbox.addWidget(self.size_reducez_btn)

        self.size_enlarge_hbox.addWidget(self.size_enlargex_btn)
        self.size_enlarge_hbox.addWidget(self.size_enlargey_btn)
        self.size_enlarge_hbox.addWidget(self.size_enlargez_btn)

        self.buttons_hsbox.addWidget(self.get_size_btn)
        self.buttons_hsbox.addWidget(self.set_size_btn)

        self.size_gbox_layout.addLayout(self.size_hbox)
        self.size_gbox_layout.addLayout(self.buttons_hsbox)
        self.size_gbox_layout.addLayout(self.size_reduce_hbox)
        self.size_gbox_layout.addLayout(self.size_enlarge_hbox)

        self.vbox.addWidget(self.pos_gbox)
        self.vbox.addWidget(self.size_gbox)

        self.setLayout(self.vbox)

        self.exec()

    def getRegionPosition(self):
        self.pos_x.setText(str(Metashape.app.document.chunk.region.center.x))
        self.pos_y.setText(str(Metashape.app.document.chunk.region.center.y))
        self.pos_z.setText(str(Metashape.app.document.chunk.region.center.z))

    def setRegionPosition(self):
        Metashape.app.document.chunk.region.center = Metashape.Vector([float(self.pos_x.text()), float(self.pos_y.text()), float(self.pos_z.text())])

    def getRegionSize(self):
        self.sz_x.setText(str(Metashape.app.document.chunk.region.size.x))
        self.sz_y.setText(str(Metashape.app.document.chunk.region.size.y))
        self.sz_z.setText(str(Metashape.app.document.chunk.region.size.z))

    def setRegionSize(self):
        Metashape.app.document.chunk.region.size = Metashape.Vector([float(self.sz_x.text()), float(self.sz_y.text()), float(self.sz_z.text())])

    def moveRegion(self, direction, subofset=Metashape.Vector([0, 0, 0])):
        old_pos = Metashape.app.document.chunk.region.center
        offset = Metashape.Vector((0, 0, 0))
        new_offset = Metashape.Vector((0, 0, 0))

        if direction == 1:
            offset = Metashape.Vector([Metashape.app.document.chunk.region.size.x, 0, 0])
            new_offset = Metashape.Matrix.Rotation(Metashape.app.document.chunk.region.rot).mulv(offset - subofset)
            Metashape.app.document.chunk.region.center = old_pos + new_offset

        if direction == 2:
            offset = Metashape.Vector([Metashape.app.document.chunk.region.size.x, 0, 0])
            new_offset = Metashape.Matrix.Rotation(Metashape.app.document.chunk.region.rot).mulv(offset - subofset)
            Metashape.app.document.chunk.region.center = old_pos - new_offset

        if direction == 3:
            offset = Metashape.Vector([0, Metashape.app.document.chunk.region.size.y, 0])
            new_offset = Metashape.Matrix.Rotation(Metashape.app.document.chunk.region.rot).mulv(offset - subofset)
            Metashape.app.document.chunk.region.center = old_pos + new_offset

        if direction == 4:
            offset = Metashape.Vector([0, Metashape.app.document.chunk.region.size.y, 0])
            new_offset = Metashape.Matrix.Rotation(Metashape.app.document.chunk.region.rot).mulv(offset - subofset)
            Metashape.app.document.chunk.region.center = old_pos - new_offset

        if direction == 5:
            offset = Metashape.Vector([0, 0, Metashape.app.document.chunk.region.size.z])
            new_offset = Metashape.Matrix.Rotation(Metashape.app.document.chunk.region.rot).mulv(offset - subofset)
            Metashape.app.document.chunk.region.center = old_pos + new_offset

        if direction == 6:
            offset = Metashape.Vector([0, 0, Metashape.app.document.chunk.region.size.z])
            new_offset = Metashape.Matrix.Rotation(Metashape.app.document.chunk.region.rot).mulv(offset - subofset)
            Metashape.app.document.chunk.region.center = old_pos - new_offset

    def enlargeRegion(self, axis):
        rs = Metashape.app.document.chunk.region.size

        if axis == 0:
            self.moveRegion(1, Metashape.Vector([rs.x / 2, 0, 0]))
            Metashape.app.document.chunk.region.size = Metashape.Vector([rs.x * 2, rs.y, rs.z])

        if axis == 1:
            self.moveRegion(3, Metashape.Vector([0, rs.y / 2, 0]))
            Metashape.app.document.chunk.region.size = Metashape.Vector([rs.x, rs.y * 2, rs.z])

        if axis == 2:
            self.moveRegion(5, Metashape.Vector([0, 0, rs.z / 2]))
            Metashape.app.document.chunk.region.size = Metashape.Vector([rs.x, rs.y, rs.z * 2])

    def reduceRegion(self, axis):
        rs = Metashape.app.document.chunk.region.size

        if axis == 0:
            Metashape.app.document.chunk.region.size = Metashape.Vector([rs.x / 2, rs.y, rs.z])
            self.moveRegion(2, Metashape.Vector([rs.x / 4, 0, 0]))

        if axis == 1:
            Metashape.app.document.chunk.region.size = Metashape.Vector([rs.x, rs.y / 2, rs.z])
            self.moveRegion(4, Metashape.Vector([0, rs.y / 4, 0]))

        if axis == 2:
            Metashape.app.document.chunk.region.size = Metashape.Vector([rs.x, rs.y, rs.z / 2])
            self.moveRegion(6, Metashape.Vector([0, 0, rs.z / 4]))


def show_region_dialog():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = ChunkRegionControl(parent)


label = "Scripts/Region Control"
Metashape.app.addMenuItem(label, show_region_dialog)
print("To execute this script press {}".format(label))
