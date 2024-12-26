import Metashape

"""
Metashape disable model Script (v 1.0)
Kent Mori, Feb 2022

Usage:
Workflow -> Batch Process -> Add -> Run script
This scrip disable current model of chunks. 
It is useful when you want to make models different quality in one batch process.
"""

compatible_major_version = "2.2"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

for chunk in chunks:
    if chunk.enabled is True:
        chunk.model = None