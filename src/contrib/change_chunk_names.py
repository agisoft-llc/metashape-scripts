import Metashape

"""
Metashape Chunk Name Changer Script (v 1.0)
Kent Mori, Feb 2021
Usage:
Workflow -> Batch Process -> Add -> Run script

This script changes chunks name refering to the first image name of the chunk.
When the image name include "_", this splits the name and join first three words.
ex) image name: "M33333_human_CR_000" -> chunk name: "M33333_human_CR"

"""

compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

for chunk in chunks:
    if chunk.enabled is True:
        camera_name = str(chunk.cameras[0].label)
        chunk_name = "_".join(camera_name.split("_")[:3])
        chunk.label = chunk_name