# Rotates model coordinate system in accordance of bounding box for active chunk. Scale is kept.
#
# This is python script for PhotoScan Pro. Scripts repository: https://github.com/agisoft-llc/photoscan-scripts

import PhotoScan
import math

# Checking compatibility
compatible_major_version = "1.4"
found_major_version = ".".join(PhotoScan.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible PhotoScan version: {} != {}".format(found_major_version, compatible_major_version))


def cs_to_bbox():
    print("Script started...")

    doc = PhotoScan.app.document
    chunk = doc.chunk

    R = chunk.region.rot     # Bounding box rotation matrix
    C = chunk.region.center  # Bounding box center vector

    if chunk.transform.matrix:
        T = chunk.transform.matrix
        s = math.sqrt(T[0, 0] ** 2 + T[0, 1] ** 2 + T[0, 2] ** 2)  # scaling # T.scale()
        S = PhotoScan.Matrix().Diag([s, s, s, 1])                  # scale matrix
    else:
        S = PhotoScan.Matrix().Diag([1, 1, 1, 1])

    T = PhotoScan.Matrix([[R[0, 0], R[0, 1], R[0, 2], C[0]],
                          [R[1, 0], R[1, 1], R[1, 2], C[1]],
                          [R[2, 0], R[2, 1], R[2, 2], C[2]],
                          [      0,       0,       0,    1]])

    chunk.transform.matrix = S * T.inv()  # resulting chunk transformation matrix

    print("Script finished!")


label = "Custom menu/Coordinate system to bounding box"
PhotoScan.app.addMenuItem(label, cs_to_bbox)
print("To execute this script press {}".format(label))
