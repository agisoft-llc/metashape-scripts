import Metashape

"""
Metashape Mesh debris Filter Script (v 1.1)
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

pow_1 = 0   #threshold power when compo_number > 10
thld_1 = 10000   #threshold
pow_2 = 1   #threshold power when compo_number =< 10
thld_2 = 10000   #basic threshold when compo_number =< 10

def removeComp(chunk):
    chunk.model.removeComponents(thld_1)
    st = chunk.model.statistics()
    print("threshold_" + str(thld_1))
    print("remaining_" + str(st.components))
    return st.components

for chunk in chunks:
    if chunk.enabled is True:
        stats = chunk.model.statistics()
        compo_number = stats.components
        print(compo_number)
        
        while compo_number != 1:
            if compo_number > 10:
                pow_1 = pow_1 + 1
                thld_1 = 1000*10**pow_1   #increase exponentially
                thld_2 = thld_1
                compo_number = removeComp(chunk)
             
            elif compo_number == 0:
                break
            
            else:
                pow_2 = pow_2 + 1
                thld_1 = thld_2 * pow_2
                compo_number = removeComp(chunk)