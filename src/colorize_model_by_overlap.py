# Colorizes model w.r.t. overlap of cameras (like in report's pdf overlap overview).
#
# Note that it doesn't respect occlusions and just calculates number of projections from each vertex to all cameras
# (without any checks for occlusions and distance between the vertex and the camera).
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
import time

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def colorize_model_vertices_by_overlap():
    print("Script started...")
    doc = Metashape.app.document
    chunk = doc.chunk
    vertices = chunk.model.vertices
    cameras = [camera for camera in chunk.cameras if camera.transform]

    start_time = time.time()

    colors = [(255, 255, 255),
              (220, 50, 50),
              (220, 125, 50),
              (220, 200, 50),
              (165, 220, 50),
              (90, 220, 50),
              (50, 220, 85),
              (50, 220, 160),
              (50, 205, 220),
              (50, 125, 220),
              (50, 50, 220)]

    nvertices = len(vertices)
    print("{} vertices and {} cameras...".format(nvertices, len(cameras)))

    logging_step = 10*1000
    for i, vert in enumerate(vertices):
        coord = vert.coord
        overlap = 0

        for camera in cameras:
            if camera.transform.inv().mulp(coord)[2] < 0:
                continue

            if 0 < camera.project(coord)[0] < camera.sensor.width:
                if 0 < camera.project(coord)[1] < camera.sensor.height:
                    overlap += 1
                    if overlap == len(colors) - 1:
                        break
        vert.color = colors[overlap]
        if (i % logging_step) == (logging_step - 1) or i == (nvertices - 1):
            print("{}% - {}/{} vertices processed".format(int((i + 1) * 100.0 / nvertices), i + 1, nvertices))
            Metashape.app.update()

    # We need to notify Metashape about changes in model, so that it will be shown with new colors:
    # let's do this via workaround - making a copy of model and then deleting the previous one
    # (just to trigger the update of Model View)
    tmp_model = chunk.model
    copy_model = tmp_model.copy()
    copy_model.label = tmp_model.label
    copy_model.bands = ['Red', 'Green', 'Blue']
    tmp_model.clear()

    Metashape.app.update()
    print("Script finished in {:.2f} seconds.".format(time.time() - start_time))


label = "Scripts/Color Model with Overlap"
Metashape.app.addMenuItem(label, colorize_model_vertices_by_overlap)
print("To execute this script press {}".format(label))
