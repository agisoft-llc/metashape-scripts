# Script save model renders for all aligned cameras to the same folder where the source photos are present with the "_render" suffix.
#
# This is python script for PhotoScan Pro. Scripts repository: https://github.com/agisoft-llc/photoscan-scripts

import PhotoScan
import os

# Checking compatibility
compatible_major_version = "1.3"
found_major_version = ".".join(PhotoScan.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible PhotoScan version: {} != {}".format(found_major_version, compatible_major_version))


def render_cameras():
    print("Script started...")

    chunk = PhotoScan.app.document.chunk
    if not chunk.model:
        raise Exception("No model!")

    for camera in chunk.cameras:
        if not camera.transform:
            continue

        render = chunk.model.renderImage(camera.transform, camera.sensor.calibration)

        photo_dir = os.path.dirname(camera.photo.path)
        photo_filename = os.path.basename(camera.photo.path)
        render_filename = os.path.splitext(photo_filename)[0] + "_render.jpg"

        render.save(os.path.join(photo_dir, render_filename))

    print("Script finished!")


label = "Custom menu/Render photos for cameras"
PhotoScan.app.addMenuItem(label, render_cameras)
print("To execute this script press {}".format(label))
