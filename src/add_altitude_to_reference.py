# Script adds user defined altitude to Source values in the Reference pane.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape

# Checking compatibility
compatible_major_version = "1.6"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def add_altitude():
    """
    Adds user-defined altitude for camera instances in the Reference pane
    """

    doc = Metashape.app.document
    if not len(doc.chunks):
        raise Exception("No chunks!")

    alt = Metashape.app.getFloat("Please specify the height to be added:", 100)

    print("Script started...")
    chunk = doc.chunk

    for camera in chunk.cameras:
        if camera.reference.location:
            coord = camera.reference.location
            camera.reference.location = Metashape.Vector([coord.x, coord.y, coord.z + alt])

    print("Script finished!")


label = "Custom menu/Add reference altitude"
Metashape.app.addMenuItem(label, add_altitude)
print("To execute this script press {}".format(label))
