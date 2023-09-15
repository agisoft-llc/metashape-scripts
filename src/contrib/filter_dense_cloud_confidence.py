import Metashape
import sys

"""
Metashape Dense Cloud Filter by Confidence (v 0.1)
Jules Fleury, SIGéo/CEREGE/AMU
Usage:
Tools -> Run script
In the row "Arguments" enter the maximum confidence level to cut
ex: 3
The script will then select and remove all points in the confidence level [0,3]
or leave the "Arguments" line blank, in which case the default value will be used.
"""

def_maxconf = 3

if len(sys.argv) == 2:
	maxconf = int(sys.argv[1])
	print("Using max confidence value from user argument " + str(maxconf) + "\n")
else:
	maxconf = def_maxconf
	print("Using max confidence value from default value " + str(maxconf) + "\n")


def filter_dense_cloud(chunk, maxconf):
	chunk.dense_cloud.setConfidenceFilter(0, maxconf)  # configuring dense cloud filter so that only point with low-confidence currently active
	all_points_classes = list(range(128))
	chunk.dense_cloud.removePoints(all_points_classes)  # removes all active points of the dense cloud, i.e. removing all low-confidence points
	chunk.dense_cloud.resetFilters()  # resetting filter, so that all other points (i.e. high-confidence points) are now active


for chunk in Metashape.app.document.chunks:
	print("Porcessing chunk " + chunk.label + "...")
	filter_dense_cloud(chunk, maxconf)
