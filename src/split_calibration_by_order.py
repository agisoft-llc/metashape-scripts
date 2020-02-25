# Script groups the cameras into calibration groups according to the camera parameters (pixel size, focal length) and picture order.
# Only group of consecutive images can be put to the same calibration group.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape

# Checking compatibility
compatible_major_version = "1.6"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def split_cameras_calibration_group_by_order():
    print("Script started...")

    chunk = Metashape.app.document.chunk 
    for camera in chunk.cameras:
        sensor = camera.sensor
        new_sensor = chunk.addSensor()

        new_sensor.calibration = sensor.calibration
        new_sensor.fixed = sensor.fixed
        new_sensor.focal_length = sensor.focal_length
        new_sensor.height = sensor.height
        new_sensor.width = sensor.width
        new_sensor.label = camera.photo.meta['Exif/Model'] + " (" + camera.photo.meta['Exif/FocalLength'] + " mm)"#sensor.label + ", " + camera.label
        new_sensor.pixel_height = sensor.pixel_height
        new_sensor.pixel_width = sensor.pixel_width
        new_sensor.pixel_size = sensor.pixel_size
        new_sensor.type = sensor.type
        new_sensor.user_calib = sensor.user_calib

        camera.sensor = new_sensor

    print("Intermediate stage completed...")    
    for i in range(1, len(chunk.cameras)):
        camera = chunk.cameras[i]
        prev = chunk.cameras[i-1]

        if (camera.sensor.width == prev.sensor.width) and (camera.sensor.height == prev.sensor.height):
            if (camera.sensor.pixel_height == prev.sensor.pixel_height) and (camera.sensor.pixel_width == prev.sensor.pixel_width):
                if camera.sensor.focal_length == prev.sensor.focal_length:
                    camera.sensor = prev.sensor

    print("Script finished, grouped calibration groups by the order.")


label = "Custom menu/Split calibration groups by order"
Metashape.app.addMenuItem(label, split_cameras_calibration_group_by_order)
print("To execute this script press {}".format(label))
