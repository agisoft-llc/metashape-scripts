# Script shows general workflow for processing laser scans with imagery.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
from packaging import version
from PySide2.QtWidgets import QMessageBox

# Checking compatibility
compatible_version = "2.0.2"
if version.parse(Metashape.app.version) < version.parse(compatible_version):
    raise Exception("Incompatible Metashape version: {} < {}".format(Metashape.app.version, compatible_version))

def process_laser_scans_with_images():
    preserve_laser_scans_relative_position = False
    preserve_laser_scans_absolute_position = False

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    doc = Metashape.app.document
    if not doc.path:
        path = Metashape.app.getSaveFileName("Save Project As:", filter = "*.psx");
        doc.save(path)
    chunk = doc.chunk

    laser_scan_paths = Metashape.app.getOpenFileNames("Select laser scans:")
    if len(laser_scan_paths) == 0:
        raise Exception("Select at least 1 laser scan")

    photo_paths = Metashape.app.getOpenFileNames("Select photos:")

    if len(laser_scan_paths) == 1:
        preserve_laser_scans_relative_position = True
    elif len(laser_scan_paths) > 1:
        ans = QMessageBox.question(
            parent, '',
            "Are laser scans prealigned to each other?",
            QMessageBox.Yes | QMessageBox.No
        )
        if QMessageBox.Yes == ans:
            preserve_laser_scans_relative_position = True

    if preserve_laser_scans_relative_position or len(laser_scan_paths) == 1:
        ans = QMessageBox.question(
            parent, '',
            "Do you want to preserve laser scans absolute position?",
            QMessageBox.Yes | QMessageBox.No
        )
        if QMessageBox.Yes == ans:
            preserve_laser_scans_absolute_position = True

    for laser_scan_path in laser_scan_paths:
        chunk.importPointCloud(laser_scan_path, is_laser_scan=True)
        doc.save()

    asset_group = chunk.addPointCloudGroup()
    doc.save()

    initial_asset_group_transform = None
    initial_asset_group_crs = None
    
    for point_cloud in chunk.point_clouds:
        point_cloud.asset_group = asset_group
    doc.save()

    if preserve_laser_scans_relative_position:
        chunk.setGroupFixed([asset_group], True)

        initial_asset_group_transform = asset_group.transform
        initial_asset_group_crs = asset_group.crs

        # unlock transform
        asset_group.crs = None
    else:
        for point_cloud in chunk.point_clouds:
            # unlock transform
            point_cloud.crs = None

    chunk.addPhotos(photo_paths)
    doc.save()

    chunk.matchPhotos(keypoint_limit = 40000, tiepoint_limit = 10000, generic_preselection = False, reference_preselection = False)
    doc.save()

    chunk.alignCameras(reset_alignment = True)
    doc.save()

    if preserve_laser_scans_absolute_position:
        chunk.crs = initial_asset_group_crs
        chunk.transform.matrix = initial_asset_group_transform * asset_group.transform.inv()

    chunk.buildDepthMaps(downscale = 2, filter_mode = Metashape.MildFiltering)
    doc.save()

    chunk.buildModel(source_data = Metashape.DepthMapsData)
    doc.save()

label = "Scripts/Process Laser Scans"
Metashape.app.addMenuItem(label, process_laser_scans_with_images)
print("To execute this script press {}".format(label))
