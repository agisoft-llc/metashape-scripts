# Creates spherical panorama from the current view point.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import math
import Metashape

view_consistent_direction = False
result_height_px = 4000
source_data = Metashape.DataSource.ModelData
# source_data = Metashape.DataSource.PointCloudData

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    print("Unsupported Metashape version: {} != {}. Script may not work properly".format(found_major_version, compatible_major_version))

def render_spherical_panorama(result_height_px, center, rotation, result_path):
    """
    Creates spherical panorama in the "center" point with the camera basis(right, up, back)
    transformed by the 3x3 "rotation" matrix.
    """

    chunk = Metashape.app.document.chunk
    result_width_px = 2 * result_height_px
    y_up_z_back = Metashape.Matrix.Scale([1.0, -1.0, -1.0])

    calibration = Metashape.Calibration()
    calibration.f = result_height_px / math.pi
    calibration.width = result_width_px
    calibration.height = result_height_px 
    calibration.type = Metashape.Sensor.Type.Spherical

    transform = chunk.transform.matrix.inv() * Metashape.Matrix.Translation(center) * Metashape.Matrix.Rotation(rotation) * y_up_z_back

    if (source_data == Metashape.DataSource.ModelData):
        chunk.model.renderImage(transform, calibration).save(result_path)
    elif (source_data == Metashape.DataSource.PointCloudData):
        chunk.point_cloud.renderImage(transform, calibration, point_size=4).save(result_path)

def render_panorama_from_current_point():

    chunk = Metashape.app.document.chunk
    if (chunk == None):
        raise Exception("Null chunk")

    if (source_data == Metashape.DataSource.ModelData):
        model = chunk.model
        if (model == None):
            raise Exception("Null model")
        if (model.getActiveTexture() == None):
            raise Exception("Model has no texture")

    if (source_data == Metashape.DataSource.PointCloudData):
        if (chunk.point_cloud == None):
            raise Exception("Null point cloud")

    result_path = Metashape.app.getSaveFileName("Resulting render path", filter="All Images (*.tif *.jpg *.png);;TIFF (*.tif);;JPEG (*.jpg);;PNG (*.png);;All Formats(*.*)");
    if (result_path == ""):
        print("No result path. Aborting")
        return

    viewpoint = Metashape.app.model_view.viewpoint
    view_center = viewpoint.center

    # z up, x front, y left
    z_up_rotation = chunk.crs.localframe(view_center).rotation().t() * Metashape.Matrix([
        [0,  0, -1],
        [-1, 0, 0 ],
        [0,  1, 0 ]
    ])

    # panorama from viewpoint direction
    view_rotation = viewpoint.rot

    if (view_consistent_direction):
        rotation = view_rotation
    else:
        rotation = z_up_rotation

    print("Started rendering...")

    render_spherical_panorama(result_height_px, view_center, rotation, result_path)
    
    print("Script finished!")

label = "Scripts/Render spherical panorama"
Metashape.app.addMenuItem(label, render_panorama_from_current_point)
print("To execute this script press {}".format(label))
