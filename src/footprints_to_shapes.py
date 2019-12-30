# Creates footprint shape layer in the active chunk.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape

# Checking compatibility
compatible_major_version = "1.6"
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

    if chunk.model:
        surface = chunk.model
    elif chunk.dense_cloud:
        surface = chunk.dense_cloud
    else:
        surface = chunk.point_cloud

    for camera in chunk.cameras:
        if camera.type != Metashape.Camera.Type.Regular or not camera.transform:
            continue  # skipping NA cameras

        sensor = camera.sensor
        corners = list()
        for i in [[0, 0], [sensor.width - 1, 0], [sensor.width - 1, sensor.height - 1], [0, sensor.height - 1]]:
            corners.append(surface.pickPoint(camera.center, camera.unproject(Metashape.Vector(i))))
            if not corners[-1]:
                corners[-1] = chunk.point_cloud.pickPoint(camera.center, camera.unproject(Metashape.Vector(i)))
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
            shape.type = Metashape.Shape.Type.Polygon
            shape.group = footprints
            shape.vertices = corners
            shape.has_z = True

    Metashape.app.update()
    print("Script finished!")


label = "Custom menu/Create footprint shape layer"
Metashape.app.addMenuItem(label, create_footprints)
print("To execute this script press {}".format(label))
