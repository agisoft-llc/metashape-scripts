# Imports DJI trajectory text file (*_sbet.txt).
# Don't confuse with *_sbet.out files produced by DJI Terra which can be imported directly into Metashape
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import os
import struct
from pathlib import Path

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def convert_txt_to_sbet(filename, dest):
    num_valid_lines = 0
    with Path(filename).open('r') as file:
        for line in file:
            try:
                tokens = line.split(' ')
                float_strings = filter(lambda s : bool(s), tokens)
                floats = map(lambda s : float(s), float_strings)
                floats = list(floats)
                assert(len(floats) == 17)

                dest.writelines([struct.pack('<ddddddddddddddddd', *floats)]);
                num_valid_lines += 1
            except:
                pass
    
    if (num_valid_lines == 0):
        raise Exception("Unable to parse %s: Expected at least one line with 17 floats sebparated by spaces" % filename)
    else:
        print("Parsed %i valid lines" % num_valid_lines)

def import_dji_text():
    doc = Metashape.app.document
    chunk = doc.chunk

    temp = tempfile.NamedTemporaryFile(mode='wb+', suffix='.sbet', delete = False)
    try:
        convert_txt_to_sbet(Metashape.app.getOpenFileName("Select DJI sbet text file", "", "*.txt"), temp)
    except:
        temp.close()
        Path(temp.name).unlink()
        raise

    temp.close()
    Metashape.app.document.chunk.importTrajectory(temp.name)
    Path(temp.name).unlink()


label = "Scripts/Import DJI trajectory_sbet.txt"
Metashape.app.addMenuItem(label, import_dji_text)
print("To execute this script press {}".format(label))
