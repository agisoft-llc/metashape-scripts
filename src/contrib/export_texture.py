from ast import Break
from importlib.resources import path
import Metashape
import sys

"""
Metashape export texture Script (v 1.0)
Kent Mori, Feb 2022

Usage:
Workflow -> Batch Process -> Add -> Run script
This scrip export textures of the chunks. 
"""

compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

def pathforsave(tex,img):   #def for save path and chooseing image type
    if tex == 1:
        a = str("_color")
    elif tex == 2:
        a = str("_normal")
    else:
        print("Texture type error")
        Break
    
    if img == 1:
        b = str(".png")
    elif img == 2:
        b = str(".jpg")
    else:
        print("Image type error")
        Break

    savePath = str("/".join(doc.path.split("/")[:-1]) + "/" + chunk.label + "_models/fromMS/" + chunk.label)
    return savePath + a + b

def texturetype(tex):   #def for choosing texture type
    if tex == 1:
        c = Metashape.Model.DiffuseMap
    elif tex == 2:
        c = Metashape.Model.NormalMap
    else:
        print("Texture type error")
        Break
    return c

if len(sys.argv) == 1:  #when no system argv, it makes textures on the whole chanks enabled.
    print("Run at whole chanks enabled")
    for chunk in chunks:
        if chunk.enabled is True:
            if chunk.model is None:
                continue
            else:
                chunk.exportTexture(path = pathforsave(1,1), texture_type=texturetype(1), save_alpha=False)
                chunk.exportTexture(path = pathforsave(1,2), texture_type=texturetype(1), save_alpha=False)
                chunk.exportTexture(path = pathforsave(2,2), texture_type=texturetype(2), save_alpha=False)

else:   #when something in the system argv, it makes textures on the just one chank selected.
    chunk = doc.chunk
    print("Run at a selected chunk")
    chunk.exportTexture(path = pathforsave(1,1), texture_type = texturetype(1), save_alpha=False)
    chunk.exportTexture(path = pathforsave(1,2), texture_type = texturetype(1), save_alpha=False)
    chunk.exportTexture(path = pathforsave(2,2), texture_type = texturetype(2), save_alpha=False)