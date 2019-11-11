import Metashape
import sys

"""
Metashape Sparse Point Cloud Filter Script (v 1.5)
Matjaz Mori, CPA, June 2019
Usage:
Workflow -> Batch Process -> Add -> Run script
In the row "Argumets" we enter exactly 4 values ​​without spaces for: ReprojectionError, ReconstructionUncertainty, ImageCount, ProjectionAccuracy in this order.
ex: 1 15 3 5
or
leave the "Arguments" line blank, in which case the default values ​​will be used,
which can also be modified in the script itself under the variables def_reperr, def_recunc, def_imgcount and def_projacc.
When using, it is advisable to monitor the Console (View -> Console).
"""


def_reperr=1
def_recunc=15
def_imgcount=3
def_projacc=5

paramNo=len(sys.argv)

reperr=float(sys.argv[1] if paramNo == 5 else def_reperr)
recunc=float(sys.argv[2] if paramNo == 5 else def_recunc)
imgcount=float(sys.argv[3] if paramNo == 5 else def_imgcount)
projacc=float(sys.argv[4] if paramNo == 5 else def_projacc)



for chunk in Metashape.app.document.chunks:
    f = Metashape.PointCloud.Filter()
    f.init(chunk,Metashape.PointCloud.Filter.ReprojectionError)
    f.removePoints(reperr)
	
for chunk in Metashape.app.document.chunks:
    f = Metashape.PointCloud.Filter()
    f.init(chunk,Metashape.PointCloud.Filter.ReconstructionUncertainty)
    f.removePoints(recunc)

for chunk in Metashape.app.document.chunks:
    f = Metashape.PointCloud.Filter()
    f.init(chunk,Metashape.PointCloud.Filter.ImageCount)
    f.removePoints(imgcount)
	
for chunk in Metashape.app.document.chunks:
    f = Metashape.PointCloud.Filter()
    f.init(chunk,Metashape.PointCloud.Filter.ProjectionAccuracy)
    f.removePoints(projacc)

if paramNo == 5:
    print ("Number of entered arguments: " +str(paramNo-1)+". Used values:")
else:
    print ("Number of entered arguments: " +str(paramNo-1)+". Default values were used:")
	
print ("ReprojectionError Level: ")
print (reperr)
print ("ReconstructionUncertainty Level: ")
print (recunc)
print ("ImageCount Level: ")
print (imgcount)
print ("ProjectionAccuracy Level: ")
print (projacc)