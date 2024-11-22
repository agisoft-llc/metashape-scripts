"""
Reverse Marker Selection
Jason Hinsley Sept. 2024
Usage:Workflow -> Batch Process -> Add -> Run script to add to menu.
Select Scripts -> Remove Unpinned Projections

This script will reverse the current marker selection.
Currently selected markers will be deselected and vise versa.
This swaps markers from GCP to CP quickly.
"""

import datetime
import Metashape

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

def revese_markers():
    print (datetime.datetime.now())

    doc = Metashape.app.document
    chunk = doc.chunk

    # are there markers present in current chunk?
    if len(chunk.markers) == 0:
        raise Exception("No markers present in current chunk")

    message = 'Reversing marker selection'
    print (message)

    # GCP: Ground Control Point - Marker used to reference the model.
    # CP: Check Point - Marker used to validate the accuracy of the camera alignment and optimization procedures results.
    for marker in chunk.markers:
        currentStatus = "GCP" if marker.reference.enabled else "Check Point"
        newStatus = "GCP" if not marker.reference.enabled else "Check Point"

        message = 'Changing marker: ' + marker.label + ' from ' + currentStatus + ' --> ' + newStatus
        print (message)

        marker.reference.enabled = not marker.reference.enabled

    message = 'Success, markers reversed!'
    print (message)

label = "Scripts/Reverse Marker Selection"

# remove menu, if exists
Metashape.app.removeMenuItem(label)

# add custom menu item
Metashape.app.addMenuItem(label, revese_markers)
print("To execute this script press {}".format(label))