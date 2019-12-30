# Created by GeoScan Ltd. (http://geoscan.aero)
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape as ps
import math, time

# Checking compatibility
compatible_major_version = "1.6"
found_major_version = ".".join(ps.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

try:
    from PySide2.QtGui import *
    from PySide2.QtCore import *
    from PySide2.QtWidgets import *
except ImportError:
    from PySide.QtGui import *
    from PySide.QtCore import *
import copy


def time_measure(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print("Finished processing in {} sec.".format(t2 - t1))
        return res

    return wrapper


def delta_vector_to_chunk(v1, v2):
    chunk = ps.app.document.chunk
    v1 = chunk.crs.unproject(v1)
    v2 = chunk.crs.unproject(v2)
    v1 = chunk.transform.matrix.inv().mulp(v1)
    v2 = chunk.transform.matrix.inv().mulp(v2)
    z = v2 - v1
    z.normalize()

    return z


def get_chunk_vectors(lat, lon):
    z = delta_vector_to_chunk(ps.Vector([lon, lat, 0]), ps.Vector([lon, lat, 1]))
    y = delta_vector_to_chunk(ps.Vector([lon, lat, 0]), ps.Vector([lon + 0.001, lat, 0]))
    x = delta_vector_to_chunk(ps.Vector([lon, lat, 0]), ps.Vector([lon, lat + 0.001, 0]))
    return x, y, -z


def wgs_to_chunk(chunk, point):
    return chunk.transform.matrix.inv().mulp(chunk.crs.unproject(point))


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


# returns distance estimation between two cameras in chunk
def get_photos_delta(chunk):
    mid_idx = int(len(chunk.cameras) / 2)
    if mid_idx == 0:
        return ps.Vector([0, 0, 0])
    c1 = chunk.cameras[:mid_idx][-1]
    c2 = chunk.cameras[:mid_idx][-2]
    print(c1.reference.location)
    print(c2.reference.location)
    offset = c1.reference.location - c2.reference.location
    for i in range(len(offset)):
        offset[i] = math.fabs(offset[i])
    return offset


def get_chunk_bounds(chunk):
    min_latitude = min(c.reference.location[1] for c in chunk.cameras if c.reference.location is not None)
    max_latitude = max(c.reference.location[1] for c in chunk.cameras if c.reference.location is not None)
    min_longitude = min(c.reference.location[0] for c in chunk.cameras if c.reference.location is not None)
    max_longitude = max(c.reference.location[0] for c in chunk.cameras if c.reference.location is not None)
    offset = get_photos_delta(chunk)
    offset_factor = 2
    delta_latitude = offset_factor * offset.y
    delta_longitude = offset_factor * offset.x

    min_longitude -= delta_longitude
    max_longitude += delta_longitude
    min_latitude -= delta_latitude
    max_latitude += delta_latitude

    return min_latitude, min_longitude, max_latitude, max_longitude


# Evaluates rotation matrices for cameras that have location
# algorithm is straightforward: we assume copter has zero pitch and roll,
# and yaw is evaluated from current copter direction
# current direction is evaluated simply subtracting location of
# current camera from the next camera location
# i and j are unit axis vectors in chunk coordinate system
# i || North
def estimate_rotation_matrices(chunk, i, j):
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
                direction = delta_vector_to_chunk(c.reference.location, next_camera.reference.location)

                cos_yaw = direction * j
                yaw = math.degrees(math.acos(cos_yaw)) + 90  # TODO not sure about this offset

                if direction * i > 0:
                    yaw = -yaw

                c.reference.rotation = ps.Vector([yaw, 0, 0])

        group_cameras[-1].reference.rotation = group_cameras[-2].reference.rotation


@time_measure
def align_cameras(chunk, min_latitude, min_longitude):
    if chunk.transform.scale is None:
        chunk.transform.scale = 1
        chunk.transform.rotation = ps.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        chunk.transform.translation = ps.Vector([0, 0, 0])

    i, j, k = get_chunk_vectors(min_latitude, min_longitude)  # i || North
    estimate_rotation_matrices(chunk, i, j)

    for c in chunk.cameras:
        if c.transform is not None:
            continue

        location = c.reference.location
        if location is None:
            continue
        chunk_coordinates = wgs_to_chunk(chunk, location)
        fi = c.reference.rotation.x + 90
        fi = math.radians(fi)
        roll = math.radians(c.reference.rotation.z)
        pitch = math.radians(c.reference.rotation.y)

        roll_mat = ps.Matrix([[1,              0,               0],
                              [0, math.cos(roll), -math.sin(roll)],
                              [0, math.sin(roll), math.cos(roll)]])
        pitch_mat = ps.Matrix([[ math.cos(pitch), 0, math.sin(pitch)],
                               [               0, 1,               0],
                               [-math.sin(pitch), 0, math.cos(pitch)]])
        yaw_mat = ps.Matrix([[math.cos(fi), -math.sin(fi), 0],
                             [math.sin(fi),  math.cos(fi), 0],
                             [           0,             0, 1]])

        r = roll_mat * pitch_mat * yaw_mat

        ii = r[0, 0] * i + r[1, 0] * j + r[2, 0] * k
        jj = r[0, 1] * i + r[1, 1] * j + r[2, 1] * k
        kk = r[0, 2] * i + r[1, 2] * j + r[2, 2] * k

        c.transform = ps.Matrix([[ii.x, jj.x, kk.x, chunk_coordinates[0]],
                                 [ii.y, jj.y, kk.y, chunk_coordinates[1]],
                                 [ii.z, jj.z, kk.z, chunk_coordinates[2]],
                                 [   0,    0,    0,                    1]])


def run_camera_alignment():
    print("Script started...")

    doc = ps.app.document
    chunk = doc.chunk

    if not check_chunk(chunk):
        return

    min_latitude, min_longitude, max_latitude, max_longitude = get_chunk_bounds(chunk)
    try:
        align_cameras(chunk, min_latitude, min_longitude)
    except Exception as e:
        print(e)

    print("Script finished!")


label = "Custom menu/Apply Vertical Camera Alignment"
ps.app.addMenuItem(label, run_camera_alignment)
print("To execute this script press {}".format(label))
