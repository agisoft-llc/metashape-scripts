# Imports photos with external depth maps.
#
# How to use:
# 1. Select images containing depth information. Image should contain depth in absolute units
# representing z value in camera coordinate system.
# 2. Select corresponding color images. Color images should have the same aspect ratio as depth images
# 3. After import is done, check depth maps are correct by toggling "Show Depth Maps" icon in the tool bar in photo view
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import os
from pathlib import Path

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

def import_external_depth():
    depth_images = Metashape.app.getOpenFileNames("Select depth images:")
    if (len(depth_images) == 0):
        raise Exception("No depth images specified")

    color_images = Metashape.app.getOpenFileNames("Select color images:")
    if (len(depth_images) != len(color_images)):
        raise Exception("Number of color images should match number of depth images")

    depth_images.sort()
    color_images.sort()

    working_directory = Path(depth_images[0]).parent
    working_directory_str = str(working_directory)
    preprocessed_directory = working_directory.joinpath("preprocessed")
    preprocessed_directory.mkdir(exist_ok = True)

    print("Saving preprocessed images to " + str(preprocessed_directory))

    Metashape.app.document.chunk.importDepthImages(filenames = depth_images, color_filenames = color_images, image_path = str(preprocessed_directory) + "/{filename}.tif")

label = "Scripts/Import external depth"
Metashape.app.addMenuItem(label, import_external_depth)
print("To execute this script press {}".format(label))
