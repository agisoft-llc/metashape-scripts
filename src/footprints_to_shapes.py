# Creates footprint shape layer in the active chunk.
#
# This is python script for PhotoScan Pro. Scripts repository: https://github.com/agisoft-llc/photoscan-scripts

import PhotoScan

# Checking compatibility
compatible_major_version = "1.4"
found_major_version = ".".join(PhotoScan.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible PhotoScan version: {} != {}".format(found_major_version, compatible_major_version))


def create_footprints():
    """
    Creates four-vertex shape for each aligned camera (footprint) in the active chunk
    and puts all these shapes to a new separate shape layer
    """

    doc = PhotoScan.app.document
    if not len(doc.chunks):
        raise Exception("No chunks!")

    print("Script started...")
    chunk = doc.chunk

    if not chunk.shapes:
        chunk.shapes = PhotoScan.Shapes()
        chunk.shapes.crs = chunk.crs
    T = chunk.transform.matrix
    footprints = chunk.shapes.addGroup()
    footprints.label = "Footprints"
    footprints.color = (30, 239, 30)

    if chunk.model:
        surface = chunk.model
    elif chunk.dense_cloud:
        surface = chunk.dense_cloud
    else:
        surface = chunk.point_cloud

    for camera in chunk.cameras:
        if not camera.transform:
            continue  # skipping NA cameras

        sensor = camera.sensor
        corners = list()
        for i in [[0, 0], [sensor.width - 1, 0], [sensor.width - 1, sensor.height - 1], [0, sensor.height - 1]]:
            corners.append(surface.pickPoint(camera.center, camera.transform.mulp(sensor.calibration.unproject(PhotoScan.Vector(i)))))
            if not corners[-1]:
                corners[-1] = chunk.point_cloud.pickPoint(camera.center, camera.transform.mulp(sensor.calibration.unproject(PhotoScan.Vector(i))))
            if not corners[-1]:
                break
            corners[-1] = chunk.crs.project(T.mulp(corners[-1]))

        if not all(corners):
            print("Skipping camera " + camera.label)
            continue

        if len(corners) == 4:
            shape = chunk.shapes.addShape()
            shape.label = camera.label
            shape.attributes["Photo"] = camera.label
            shape.type = PhotoScan.Shape.Type.Polygon
            shape.group = footprints
            shape.vertices = corners
            shape.has_z = True

    PhotoScan.app.update()
    print("Script finished!")


label = "Custom menu/Create footprint shape layer"
PhotoScan.app.addMenuItem(label, create_footprints)
print("To execute this script press {}".format(label))
