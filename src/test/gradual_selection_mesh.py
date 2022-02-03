import Metashape

"""
Metashape Mesh debris Filter Script (v 1.0)
Kent Mori, Feb 2022

Usage:
Workflow -> Batch Process -> Add -> Run script
This script scans the number of components in a model and reduceing them continuously to 1 (by force).
I wanted to make "grasdual selection" tool, but this is slower than that.
"""

compatible_major_version = "1.8"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

doc = Metashape.app.document
chunks = doc.chunks

for chunk in chunks:
    if chunk.enabled is True:
        stats = chunk.model.statistics()
        cp = stats.components
        print(cp)
        i = 0   #threshold power when cp > 2
        j = 1   #threshold
        k = 1   #threshold power when cp == 2
        while cp != 1:
            if cp > 2:
                i = i + 1
                j = 10**i   #increase exponentially
                chunk.model.removeComponents(j)
                stats = chunk.model.statistics()
                cp = stats.components
                print("threshold_" + str(j))
                print("remaining_" + str(cp))
            elif cp == 2:
                k = k + 1
                j = j * k
                chunk.model.removeComponents(j)
                stats = chunk.model.statistics()
                cp = stats.components
                print("threshold_" + str(j))
                print("remaining_" + str(cp))
            elif cp == 0:
                break