# Created by GeoScan Ltd. (http://geoscan.aero)
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape as ps
import math, time

# Checking compatibility
compatible_major_version = "1.7"
found_major_version = ".".join(ps.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
import copy


def time_measure(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print("Finished processing in {} sec.".format(t2 - t1))
        return res

    return wrapper


def show_message(msg):
    msgBox = QMessageBox()
    print(msg)
    msgBox.setText(msg)
    msgBox.exec()


def check_chunk(chunk):
    if chunk is None or len(chunk.cameras) == 0:
        show_message("Empty chunk!")
        return False

    if chunk.crs is None:
        show_message("Initialize chunk coordinate system first")
        return False

    return True


def get_antenna_transform(sensor):
    location = sensor.antenna.location
    if location is None:
        location = sensor.antenna.location_ref
    rotation = sensor.antenna.rotation
    if rotation is None:
        rotation = sensor.antenna.rotation_ref
    return ps.Matrix.Diag((1, -1, -1, 1)) * ps.Matrix.Translation(location) * ps.Matrix.Rotation(ps.Utils.ypr2mat(rotation))


def init_chunk_transform(chunk):
    if chunk.transform.scale is not None:
        return
    chunk_origin = ps.Vector([0, 0, 0])
    for c in chunk.cameras:
        if c.reference.location is None:
            continue
        chunk_origin = chunk.crs.unproject(c.reference.location)
        break

    chunk.transform.scale = 1
    chunk.transform.rotation = ps.Matrix.Diag((1, 1, 1))
    chunk.transform.translation = chunk_origin


# Evaluates rotation matrices for cameras that have location
# algorithm is straightforward: we assume copter has zero pitch and roll,
# and yaw is evaluated from current copter direction
# current direction is evaluated simply subtracting location of
# current camera from the next camera location
def estimate_rotation_matrices(chunk):
    groups = copy.copy(chunk.camera_groups)

    groups.append(None)
    for group in groups:
        group_cameras = list(filter(lambda c: c.group == group, chunk.cameras))

        if len(group_cameras) == 0:
            continue

        if len(group_cameras) == 1:
            if group_cameras[0].reference.rotation is None:
                group_cameras[0].reference.rotation = ps.Vector([0, 0, 0])
            continue

        for idx, c in enumerate(group_cameras[0:-1]):
            next_camera = group_cameras[idx + 1]

            if c.reference.rotation is None:
                if c.reference.location is None or next_camera.reference.location is None:
                    continue

                prev_location = chunk.crs.unproject(c.reference.location)
                next_location = chunk.crs.unproject(next_camera.reference.location)

                direction = chunk.crs.localframe(prev_location).mulv(next_location - prev_location)

                yaw = math.degrees(math.atan2(direction.y, direction.x)) + 90
                if yaw < 0:
                    yaw = yaw + 360

                c.reference.rotation = ps.Vector([yaw, 0, 0])

        group_cameras[-1].reference.rotation = group_cameras[-2].reference.rotation


@time_measure
def align_cameras(chunk):
    init_chunk_transform(chunk)

    estimate_rotation_matrices(chunk)

    for c in chunk.cameras:
        if c.transform is not None:
            continue

        location = c.reference.location
        if location is None:
            continue

        rotation = c.reference.rotation
        if rotation is None:
            continue

        location = chunk.crs.unproject(location)  # location in ECEF
        rotation = chunk.crs.localframe(location).rotation().t() * ps.Utils.euler2mat(rotation, chunk.euler_angles) # rotation matrix in ECEF

        transform = ps.Matrix.Translation(location) * ps.Matrix.Rotation(rotation)
        transform = chunk.transform.matrix.inv() * transform * get_antenna_transform(c.sensor).inv()

        c.transform = ps.Matrix.Translation(transform.translation()) * ps.Matrix.Rotation(transform.rotation())


def run_camera_alignment():
    print("Script started...")

    doc = ps.app.document
    chunk = doc.chunk

    if not check_chunk(chunk):
        return

    try:
        align_cameras(chunk)
    except Exception as e:
        print(e)

    print("Script finished!")


label = "Custom menu/Apply Vertical Camera Alignment"
ps.app.addMenuItem(label, run_camera_alignment)
print("To execute this script press {}".format(label))
