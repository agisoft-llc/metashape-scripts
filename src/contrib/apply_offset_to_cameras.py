# Script allows users to add an offset in three directions (XYZ) to ALL
# the cameras in the Reference Panel.
#
# Note that this scripts does not apply any check to the inputs and adds an
# arbitrary value to the camera coordinates, regardless of the Coordinate
# Reference System of the Project/Cameras.
# e.g., if the CRS is a Local System or a Cartographic Reference System,
# one may provide an offset in meters, respectively for the East, North
# and altitude direction.
# if the CRS is WGS84, one has to provide the shift in degreees for
# latitude and longitude and meters for altitude.
#
# This script may be useful when one has to correct (e.g., due to a base shift)
# the coordinates of a UAV camera acquired by RTK and saved in image exif
#
# Author: Francesco Ioli (Politecnico di Milano), francesco.ioli@polimi.it
#  01/09/2022


import Metashape

print("Script started...")

# Checking compatibility
compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(
        found_major_version, compatible_major_version))


def get_input(axis_name):
    offset = Metashape.app.getFloat(
        "Please specify offset value for axis {}:".format(axis_name), 0.)
    return offset


def apply_xyz_offset():
    doc = Metashape.app.document
    chunk = doc.chunk

    if not len(doc.chunks):
        raise Exception("No chunks!")

    only_selected = False
    if len([c for c in Metashape.app.document.chunk.cameras if c.selected]) > 0:
        # if at least one camera is selected - apply offset only to selected cameras
        only_selected = True
        print("cameras selection detected - applying offset only to selected cameras...")

    offset_x = get_input("X")
    if offset_x is None:
        return
    offset_y = get_input("Y")
    if offset_y is None:
        return
    offset_z = get_input("Z")
    if offset_z is None:
        return

    ncameras = 0
    for camera in chunk.cameras:
        if only_selected and not camera.selected:
            continue
        if camera.reference.location:
            coord = camera.reference.location
            camera.reference.location = Metashape.Vector(
                [coord.x + offset_x, coord.y + offset_y, coord.z + offset_z])
            ncameras += 1
    print("Offset dx={}, dy={}, dz={} applied to {} cameras successfully".format(offset_x, offset_y, offset_z, ncameras))


label = "Scripts/Add reference offset"
Metashape.app.addMenuItem(label, apply_xyz_offset)
print("To execute this script press {}".format(label))
