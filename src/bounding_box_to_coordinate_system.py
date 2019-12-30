# Rotates chunk's bounding box in accordance of coordinate system for active chunk. Bounding box size is kept.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
import math

# Checking compatibility
compatible_major_version = "1.6"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def bbox_to_cs():
    print("Script started...")

    doc = Metashape.app.document
    chunk = doc.chunk

    T = chunk.transform.matrix

    v_t = T.mulp(Metashape.Vector([0, 0, 0]))

    if chunk.crs:
        m = chunk.crs.localframe(v_t)
    else:
        m = Metashape.Matrix().Diag([1, 1, 1, 1])

    m = m * T
    s = math.sqrt(m[0, 0] ** 2 + m[0, 1] ** 2 + m[0, 2] ** 2)  # scale factor # s = m.scale()
    R = Metashape.Matrix([[m[0, 0], m[0, 1], m[0, 2]],
                          [m[1, 0], m[1, 1], m[1, 2]],
                          [m[2, 0], m[2, 1], m[2, 2]]])
    # R = m.rotation()

    R = R * (1. / s)

    reg = chunk.region
    reg.rot = R.t()
    chunk.region = reg

    print("Script finished!")


label = "Custom menu/Bounding box to coordinate system"
Metashape.app.addMenuItem(label, bbox_to_cs)
print("To execute this script press {}".format(label))
