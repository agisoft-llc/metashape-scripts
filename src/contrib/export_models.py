import Metashape

"""
Metashape disable model Script (v 1.0)
Kent Mori, Feb 2022

Usage:
Workflow -> Batch Process -> Add -> Run script
This scrip export whole models of chunks. 
"""

compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

for chunk in chunks:
    if chunk.enabled is True:
        chunk.exportModel(path = "/".join(doc.path.split("/")[:-1]) + "/" + chunk.label + "_models/" + chunk.label + ".obj",
        texture_format=Metashape.ImageFormat.ImageFormatPNG)
# export each quarity models of each chunks
# chunk.label + quarity["high" or "low"] + ".obj"