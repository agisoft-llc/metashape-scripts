import Metashape
import pprint

compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

for chunk in chunks:
    if chunk.enabled is True:
        print(str(chunk))
        camera = chunk.cameras[-1]
        s = chunk.transform.scale
        print("scale is "+str(s))
        print("the origin is "+str(origin))

        if s != None:
            origin = (-1) * camera.center
            
            if origin != None:
                R = Metashape.Matrix().Rotation(camera.transform.rotation()* Metashape.Matrix().Diag([1, -1, 1]))
                origin = R.inv().mulp(origin)
                chunk.transform.matrix = Metashape.Matrix([[s, 0, 0, 0], [0, 0, s, 0], [0, s, 0, 0], [0, 0, 0, 1]]) * Metashape.Matrix().Translation(origin) * R.inv()
            else:
                continue