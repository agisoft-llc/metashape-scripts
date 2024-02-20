# Colorizes model w.r.t. altitude of its vertices (like DEM colors).
#
# Script requires model with vertex colors. Run Tools/Model/Colorize Vertices... before using this script.
#
# We are grateful to Sébastien Poudroux for this script :)
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def calculate_color(z, max, min):
    thresh = [(max + 2 * min) / 3, (max + min) / 2, (2 * max + min) / 3, max]

    if z < thresh[0]:
        r = 0
        g = (z - min) / (thresh[0] - min) * 255
        b = 255
        color = (r, g, b)
    elif z < thresh[1]:
        r = 0
        g = 255
        b = (thresh[1] - z) / (thresh[1] - thresh[0]) * 255
        color = (r, g, b)
    elif z < thresh[2]:
        r = (z - thresh[1]) / (thresh[2] - thresh[1]) * 255
        g = 255
        b = 0
        color = (r, g, b)
    else:
        r = 255
        g = (thresh[3] - z) / (thresh[3] - thresh[2]) * 255
        b = 0
        color = (r, g, b)

    color = tuple(map(int, color))  # convert floats to ints
    return color


def colorize_model_vertices_by_altitude():
    print("Script started...")
    doc = Metashape.app.document
    chunk = doc.chunk
    num = len(chunk.model.vertices)

    if (num > 0 && chunk.model.vertices[0].color is None):
        raise Exception("Run Tools/Model/Colorize Vertices... before this script")

    min, max = 5.1E9, -5.1E9

    if not chunk.crs:  # Local coordinates
        print("Script only works correctly for real coordinate systems")
        print("Performing script for local coordinates")

        for i in range(0, num):
            p = chunk.model.vertices[i]

            vt = chunk.transform.matrix.mulp(p.coord)

            if vt.z > max:
                max = vt.z
            else:
                if vt.z < min:
                    min = vt.z

        for i in range(0, num):
            p = chunk.model.vertices[i]

            vt = chunk.transform.matrix.mulp(p.coord)

            p.color = calculate_color(vt.z, max, min)

    else:  # projected
        proj = chunk.crs

        for i in range(0, num):
            p = chunk.model.vertices[i]

            vt = chunk.transform.matrix.mulp(p.coord)

            vt = proj.project(vt)

            if vt.z > max:
                max = vt.z
            else:
                if vt.z < min:
                    min = vt.z

        for i in range(0, num):
            p = chunk.model.vertices[i]

            vt = chunk.transform.matrix.mulp(p.coord)
            vt = proj.project(vt)

            p.color = calculate_color(vt.z, max, min)

    # We need to notify Metashape about changes in model, so that it will be shown with new colors:
    # let's do this via workaround - making a copy of model and then deleting the previous one
    # (just to trigger the update of Model View)
    tmp_model = chunk.model
    copy_model = tmp_model.copy()
    copy_model.label = tmp_model.label
    copy_model.bands = ['Red', 'Green', 'Blue']
    tmp_model.clear()

    Metashape.app.update()
    print("Script finished.")


label = "Scripts/Color Model with Altitude"
Metashape.app.addMenuItem(label, colorize_model_vertices_by_altitude)
print("To execute this script press {}".format(label))
