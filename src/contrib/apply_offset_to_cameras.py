# Script allows users to add an offset in three directions (XYZ) to ALL
# the cameras in the Reference Panel.
#
# Note that this scripts does not apply any check to the inputs and adds an
# arbitrary value to the camera coordinates, regardless of the Coordinate
# Reference System of the Project/Cameras.
# e.g., if the CRS is a Local System or a Cartographic Reference System,
# one may provide an offset in meters, respectively for the East, North
# and altitude direction.
# if the CRS is WGS84, one has to to provide the shift in degreees for
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


def get_input():
    offset = Metashape.app.getFloat(
        "Please specify offset value:", 0.)
    return offset


def apply_x_offset():
    doc = Metashape.app.document
    chunk = doc.chunk

    if not len(doc.chunks):
        raise Exception("No chunks!")

    offset = get_input()

    for camera in chunk.cameras:
        if camera.reference.location:
            coord = camera.reference.location
            camera.reference.location = Metashape.Vector(
                [coord.x + offset, coord.y, coord.z])
    print("Offset applied successfully.")


def apply_y_offset():
    doc = Metashape.app.document
    chunk = doc.chunk

    if not len(doc.chunks):
        raise Exception("No chunks!")

    offset = get_input()

    for camera in chunk.cameras:
        if camera.reference.location:
            coord = camera.reference.location
            camera.reference.location = Metashape.Vector(
                [coord.x, coord.y + offset, coord.z])
    print("Offset applied successfully.")


def apply_z_offset():
    doc = Metashape.app.document
    chunk = doc.chunk

    if not len(doc.chunks):
        raise Exception("No chunks!")

    offset = get_input()

    for camera in chunk.cameras:
        if camera.reference.location:
            coord = camera.reference.location
            camera.reference.location = Metashape.Vector(
                [coord.x, coord.y, coord.z + offset])
    print("Offset applied successfully.")


Metashape.app.addMenuItem("Cam_offset/Add reference X", apply_x_offset)
Metashape.app.addMenuItem("Cam_offset/Add reference Y", apply_y_offset)
Metashape.app.addMenuItem("Cam_offset/Add reference Z", apply_z_offset)

print("To execute this script select an item in the 'Cam_offset' menu.")
