# Creates footprint shape layer in the active chunk.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
import multiprocessing
import concurrent.futures

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def create_footprints():
    """
    Creates four-vertex shape for each aligned camera (footprint) in the active chunk
    and puts all these shapes to a new separate shape layer
    """

    doc = Metashape.app.document
    if not len(doc.chunks):
        raise Exception("No chunks!")

    print("Script started...")
    chunk = doc.chunk

    if not chunk.shapes:
        chunk.shapes = Metashape.Shapes()
        chunk.shapes.crs = chunk.crs
    T = chunk.transform.matrix
    footprints = chunk.shapes.addGroup()
    footprints.label = "Footprints"
    footprints.color = (30, 239, 30)

    if chunk.elevation:
        surface = chunk.elevation
    elif chunk.model:
        surface = chunk.model
    elif chunk.point_cloud:
        surface = chunk.point_cloud
    else:
        surface = chunk.tie_points

    chunk_crs = chunk.crs.geoccs
    if chunk_crs is None:
        chunk_crs = Metashape.CoordinateSystem('LOCAL')

    def process_camera(chunk, camera):
        if camera.type != Metashape.Camera.Type.Regular or not camera.transform:
            return  # skipping NA cameras

        sensor = camera.sensor
        corners = list()
        for (x, y) in [[0, 0], [sensor.width - 1, 0], [sensor.width - 1, sensor.height - 1], [0, sensor.height - 1]]:
            ray_origin = camera.unproject(Metashape.Vector([x, y, 0]))
            ray_target = camera.unproject(Metashape.Vector([x, y, 1]))
            if type(surface) == Metashape.Elevation:
                dem_origin = T.mulp(ray_origin)
                dem_target = T.mulp(ray_target)
                dem_origin = Metashape.OrthoProjection.transform(dem_origin, chunk_crs, surface.projection)
                dem_target = Metashape.OrthoProjection.transform(dem_target, chunk_crs, surface.projection)
                corner = surface.pickPoint(dem_origin, dem_target)
                if corner:
                    corner = Metashape.OrthoProjection.transform(corner, surface.projection, chunk_crs)
                    corner = T.inv().mulp(corner)
            else:
                corner = surface.pickPoint(ray_origin, ray_target)
            if not corner:
                corner = chunk.tie_points.pickPoint(ray_origin, ray_target)
            if not corner:
                break
            corner = chunk.crs.project(T.mulp(corner))
            corners.append(corner)

        if len(corners) == 4:
            shape = chunk.shapes.addShape()
            shape.label = camera.label
            shape.attributes["Photo"] = camera.label
            shape.group = footprints
            shape.geometry = Metashape.Geometry.Polygon(corners)
        else:
            print("Skipping camera " + camera.label)

    with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
        executor.map(lambda camera: process_camera(chunk, camera), chunk.cameras)

    Metashape.app.update()
    print("Script finished!")


label = "Scripts/Create footprint shape layer"
Metashape.app.addMenuItem(label, create_footprints)
print("To execute this script press {}".format(label))
