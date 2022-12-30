import Metashape

"""
Metashape Chunk Name Changer Script (v 1.0)
Kent Mori, Feb 2021
Usage:
Workflow -> Batch Process -> Add -> Run script

This script saves each chunks separately.

"""

compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

for chunk in chunks:
    if chunk.enabled is True:
        doc.save(path = "/".join(doc.path.split("/")[:-1]) + "/" + chunk.label + "_chunks/" + chunk.label + ".psx", chunks = [chunk])