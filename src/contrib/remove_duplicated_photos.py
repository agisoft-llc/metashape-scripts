import datetime
import Metashape


"""
Script for removing duplicated photos, Metashape (v 1.8)
Matjaz Mori, CPA, May 2022
The script will remove all duplicated photos (photos referenced to the same file) from Metashape project. 
"""

compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def remove_duplicated_photos():
    print (datetime.datetime.now())

    doc = Metashape.app.document
    chunk = doc.chunk
    lenght = len(chunk.cameras)

    message = 'Removing duplicates...'
    print (message)

    paths = set()
    photos = list()
    for camera in list(chunk.cameras):
        if not camera.type == Metashape.Camera.Type.Regular: #skip camera track, if any
            continue

        if camera.photo.path in paths:
            photos.append(camera)
        else:
            paths.add(camera.photo.path)

    chunk.remove(photos)
    lenght_after = len(chunk.cameras)
    nr_removed = lenght-lenght_after
    message_end = 'Success, ' + str(nr_removed) + ' cameras removed.'
    print (message_end)

label = "Custom menu/Remove duplicated photos"
Metashape.app.addMenuItem(label, remove_duplicated_photos)
print("To execute this script press {}".format(label))
