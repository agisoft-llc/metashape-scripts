"""
Remove Low Quality Images (v 1.0)
Persa Koutsouradi, March 2023
Modified by Jason Hinsley Sept. 2024
Usage:Workflow -> Batch Process -> Add -> Run script to add to menu.
Select Scripts -> Remove low quality photos

This script estimates the quality of the images in a chunk.
When the image quality is less than 0.7, the image is removed form the chunk.
"""

import datetime
import Metashape

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

def remove_low_quality_photos():
    print (datetime.datetime.now())

    doc = Metashape.app.document
    chunk = doc.chunk
    length = len(chunk.cameras)

    message = 'Evaluating quality of' + str(length) + ' images...'
    print (message)

    # Estimate image quality
    chunk.analyzeImages(chunk.cameras)

    message = 'Removing low quality photos...'
    print (message)

    photos = list()

    for camera in list(chunk.cameras):
        if 'Image/Quality' in camera.meta:
            quality = float(camera.meta['Image/Quality'])

        # Cameras with quality less than 0.5 are considered blurred
        if quality < 0.7:
            photos.append(camera)

    message = 'Result: ' + str(len(photos)) + ' of ' + str(length) + ' photos are low quality!'
    print (message)

    # remove low quality images
    chunk.remove(photos)

    length_after = len(chunk.cameras)
    nr_removed = length-length_after

    message_end = 'Success, ' + str(nr_removed) + ' photos removed.'
    print (message_end)

label = "Scripts/Remove low quality photos"

# remove menu, if exists
Metashape.app.removeMenuItem(label)

# add custom menu item
Metashape.app.addMenuItem(label, remove_low_quality_photos)
print("To execute this script press {}".format(label))