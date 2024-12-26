"""
Remove Unpinned Projections
Jason Hinsley Sept. 2024
Usage:Workflow -> Batch Process -> Add -> Run script to add to menu.
Select Scripts -> Remove Unpinned Projections

This script will check that markers exist in the current chunk and remove unpinned projections (blue flags) from those markers.
Once unpinned projections are removed, update transform will be executed to make the changes take effect.
Blue (unpinned) markers usually are related to the automatically placed projections, green (pinned) markers are those that are adjusted or placed by user manually.
Both blue and green markers are considered during optimization. 
"""

import datetime
import Metashape

# Checking compatibility
compatible_major_version = "2.2"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

def remove_unpinned_projections():
    print (datetime.datetime.now())

    doc = Metashape.app.document
    chunk = doc.chunk

    # are there markers present in current chunk?
    if len(chunk.markers) == 0:
        raise Exception("No markers present in current chunk")

    message = 'Removing unpinned projections from photos...'
    print (message)

    for marker in chunk.markers:
        for camera in list(marker.projections.keys()):
            #if camera has unpinned projections
            if not marker.projections[camera].pinned:
                #remove projections from camera
                marker.projections[camera] = None

    message = 'Unpinned projections removed.'
    print (message)

    # update transform to account for removed projections
    chunk.updateTransform()

    message = 'Updating Transform...'
    print (message)

    message = 'Success, unpinned projections removed, transform updated.'
    print (message)

label = "Scripts/Remove Unpinned Projections"

# remove menu, if exists
Metashape.app.removeMenuItem(label)

# add custom menu item
Metashape.app.addMenuItem(label, remove_unpinned_projections)
print("To execute this script press {}".format(label))