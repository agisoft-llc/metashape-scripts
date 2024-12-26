import Metashape
import math

"""
Metashape Mesh debris Filter Script (v 1.1)
Kent Mori, Feb 2022

Usage:
Workflow -> Batch Process -> Add -> Run script
This script scans the number of components in a model and reduceing them continuously to 1 (by force).
I wanted to make "grasdual selection" tool, but this is slower than that.
"""

compatible_major_version = "2.2"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

def removeSmallComponents(model, faces_threshold):
    model.removeComponents(faces_threshold)
    stats = model.statistics()
    print("After removing small components with faces_threshold={}: {} faces in {} components left"
          .format(faces_threshold, stats.faces, stats.components))
    return stats

for chunk in chunks:
    if chunk.enabled is True:
        stats = chunk.model.statistics()
        print("Model has {} faces in {} components".format(stats.faces, stats.components))
        
        while stats.components > 1:
            component_avg_faces = math.ceil(stats.faces / stats.components)
            # the largest component is for sure bigger then average component faces number, so we will filter only small components
            # probably there are some small components left - so we will continue our while-loop if needed
            faces_threshold = component_avg_faces
            new_stats = removeSmallComponents(chunk.model, faces_threshold)

            assert(new_stats.components < stats.components) # checking that we deleted at least the smallest something (to ensure that script works fine)
            assert(new_stats.components > 0) # checking that the largest component is still there (to ensure that script works fine)
            stats = new_stats
