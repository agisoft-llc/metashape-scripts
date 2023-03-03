"""
Metashape Image quality estimator Script (v 1.0)
Persa Koutsouradi, March 2023
Usage:Workflow -> Batch Process -> Add -> Run script
This script estimates the quality of the images in a chunk.
When the image quality is less than 0.7, the image is removed form the chunk.

"""

import Metashape

# Checking compatibility
compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

# Calculate the image quality in the working chunk 
doc = Metashape.app.document
chunk = doc.chunk

quality_list = []

for camera in chunk.cameras:
chunk.analyzePhotos(camera)
if 'Image/Quality' in camera.meta:
quality = float(camera.meta['Image/Quality'])
if quality < 0.7:
chunk.remove(camera)
else:
quality_list.append([camera.label, quality])

# Print the quality values for verification
print("Image quality values:")
print(quality_list)
