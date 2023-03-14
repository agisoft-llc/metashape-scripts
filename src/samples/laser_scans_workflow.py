# Script shows general workflow for processing laser scans with imagery.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
import os, sys

# Checking compatibility
compatible_major_version = "2.0"
compatible_micro_version = 2
version_split = Metashape.app.version.split('.')
found_major_version = ".".join(version_split[:2])
found_micro_version = int(version_split[2])

if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))
if found_micro_version < compatible_micro_version:
    raise Exception("Incompatible Metashape version: {}.{} < {}.{}".format(found_major_version, found_micro_version, compatible_major_version, compatible_micro_version))

def find_files(folder, types):
    return [entry.path for entry in os.scandir(folder) if (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

valid_flags = ["--fix-relative", "--fix-absolute"]
folders = list(filter(lambda arg : arg[0] != "-", sys.argv[1:]))
flags   = list(filter(lambda arg : arg[0] == "-", sys.argv[1:]))
flags_valid = all(map(lambda arg : arg in valid_flags, flags))

if len(folders) != 3 or not flags_valid:
    print("Usage: general_workflow.py [--fix-relative] [--fix-absolute] <laser_scans_folder> <images_folder> <output_folder>")
    print("  --fix-relative        preserve laser scans relative position")
    print("  --fix-absolute        preserve laser scans absolute position")
    sys.exit(1)

laser_scans_folder = folders[0]
images_folder = folders[1]
output_folder = folders[2]

preserve_laser_scans_absolute_position = "--fix-absolute" in sys.argv
preserve_laser_scans_relative_position = "--fix-relative" in sys.argv or preserve_laser_scans_absolute_position

photos = find_files(images_folder, [".jpg", ".jpeg", ".tif", ".tiff"])
laser_scans = find_files(laser_scans_folder, [".e57", ".ptx"])

doc = Metashape.Document()
doc.save(output_folder + '/project.psx')

chunk = doc.addChunk()

for laser_scan_path in laser_scans:
    chunk.importPointCloud(laser_scan_path, is_laser_scan=True)
    doc.save()
laser_scan_cameras = chunk.cameras

group = chunk.addPointCloudGroup()
initial_group_crs = None
for point_cloud in chunk.point_clouds:
    point_cloud.group = group
doc.save()

if preserve_laser_scans_relative_position:
    group.fixed = True
    initial_group_crs = group.crs

    # unlock transform
    group.crs = None
else:
    for point_cloud in chunk.point_clouds:
        # unlock transform
        point_cloud.crs = None

chunk.addPhotos(photos)
if preserve_laser_scans_absolute_position:
    chunk.crs = initial_group_crs
    for cam in chunk.cameras:
        if cam not in laser_scan_cameras:
            cam.reference.enabled = False
doc.save()

chunk.matchPhotos(keypoint_limit = 40000, tiepoint_limit = 10000, generic_preselection = False, reference_preselection = False)
doc.save()

chunk.alignCameras(reset_alignment = not preserve_laser_scans_absolute_position)
doc.save()

chunk.buildDepthMaps(downscale = 2, filter_mode = Metashape.MildFiltering)
doc.save()

chunk.buildModel(source_data = Metashape.DepthMapsData)
doc.save()

if chunk.model:
    chunk.exportModel(output_folder + '/model.obj')

