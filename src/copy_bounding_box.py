# Resizes all bounding boxes to Active chunk bounding box.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape

# Checking compatibility
compatible_major_version = "1.5"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def copy_bbox():
    print("Script started...")

    doc = Metashape.app.document

    chunk = doc.chunk
    T0 = chunk.transform.matrix

    region = chunk.region
    R0 = region.rot
    C0 = region.center
    s0 = region.size

    for chunk in doc.chunks:

        if chunk == doc.chunk:
            continue

        T = chunk.transform.matrix.inv() * T0

        R = Metashape.Matrix([[T[0, 0], T[0, 1], T[0, 2]],
                              [T[1, 0], T[1, 1], T[1, 2]],
                              [T[2, 0], T[2, 1], T[2, 2]]])

        scale = R.row(0).norm()
        R = R * (1 / scale)

        region.rot = R * R0
        c = T.mulp(C0)
        region.center = c
        region.size = s0 * scale / 1.

        chunk.region = region

    print("Script finished!")


label = "Custom menu/Copy bounding box"
Metashape.app.addMenuItem(label, copy_bbox)
print("To execute this script press {}".format(label))
