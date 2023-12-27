import datetime
import shutil
import Metashape
import os
import sys
from pathlib import Path

"""
Script for moving disabled photos, Metashape (v 1.7)
Matjaz Mori, CPA, October 2019

The script will create a new subdirectory in the photos directory,
move all the photos from the project marked "Disabled" into it and remove "Disabled" cameras prom Metashape project.
When using, it is advisable to monitor the Console (View -> Console). 

"""

compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def remove_disabled_photos():
    print (datetime.datetime.now())

    doc = Metashape.app.document
    chunk = doc.chunk
    counter = 0
    counter_fail = 0
    counter_not_moved = 0
    counter_errors = 0
    counter_cameras = 0
    lenght = len(chunk.cameras)

    message = 'Starting to evaluate ' + str(lenght) + ' photos...'
    print (message)

    for camera in chunk.cameras:
        if not camera.type == Metashape.Camera.Type.Regular: #skip camera track, if any
            continue

        if camera.enabled is True:
            counter_not_moved = counter_not_moved + 1
            continue # skipping enabled cameras

        photo_path = Path(camera.photo.path)
        photo_name = str(camera.label)
        destination_dir = photo_path.parent / 'Disabled'
        destination = destination_dir / photo_path.name

        if not destination_dir.exists():
            try:
                destination_dir.mkdir()
                print ("Successfully created the directory %s " % destination_dir)
            except OSError:
                print ('Error creating %s' % destination_dir)
                counter_errors = counter_errors + 1
                continue # we can't create directory - thus we can't move photo - thus we shouldn't delete it

        try:
            if photo_path.is_file():
                print ('Moving %s ...' % photo_name)
                shutil.move(str(photo_path), str(destination))

                counter = counter + 1
                counter_cameras = counter_cameras + 1
            else:
                print ('Photo %s does not exist!' % photo_name)
                counter_cameras = counter_cameras + 1
                counter_fail = counter_fail + 1

            chunk.remove(camera)

        except OSError:
            counter_errors = counter_errors + 1
            print ('Error %s!' % photo_name)

    message_end = 'Success, ' + str(counter) + ' photos moved, ' + str(counter_not_moved) + ' photos not moved.\nNumber of files unable to move: ' + str(counter_fail) + '\nNumber of cameras removed: ' + str(counter_cameras) + '\nNumber of unknown errorrs: '+ str(counter_errors)
    print (message_end)


label = "Scripts/Remove disabled photos"
Metashape.app.addMenuItem(label, remove_disabled_photos)
print("To execute this script press {}".format(label))
