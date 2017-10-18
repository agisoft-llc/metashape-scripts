# Script adds user defined altitude to Source values in the Reference pane
#
# This is python script for PhotoScan Pro. Scripts repository: https://github.com/agisoft-llc/photoscan-scripts

import PhotoScan

# Checking compatibility
compatible_major_version = "1.3"
found_major_version = ".".join(PhotoScan.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible PhotoScan version: {} != {}".format(found_major_version, compatible_major_version))

def addAltitude():
	"""
	Adds user-defined altitude for camera instances in the Reference pane
	"""
    
	doc = PhotoScan.app.document
	if not len(doc.chunks):
		raise Exception("No chunks!")
		
	alt = PhotoScan.app.getFloat("Please specify the height to be added:", 100)
    print("Script started...")
    chunk = doc.chunk

    for camera in chunk.cameras:
        if camera.reference.location:
            coord = camera.reference.location
            camera.reference.location = PhotoScan.Vector([coord.x, coord.y, coord.z + alt])
		
    print("Script finished")
	
label = "Custom menu/Add reference altitude"
PhotoScan.app.addMenuItem(label, addAltitude)
print("To execute this script press {}".format(label))