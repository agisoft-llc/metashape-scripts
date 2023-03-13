import Metashape
import os, sys, time

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

def find_files(folder, types):
    return [entry.path for entry in os.scandir(folder) if (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

if len(sys.argv) < 3:
    print("Usage: general_workflow.py <image_folder> <output_folder>")
    sys.exit(1)

image_folder = sys.argv[1]
output_folder = sys.argv[2]

photos = find_files(image_folder, [".jpg", ".jpeg", ".tif", ".tiff"])

doc = Metashape.Document()
doc.save(output_folder + '/project.psx')

chunk = doc.addChunk()

chunk.addPhotos(photos)
doc.save()

print(str(len(chunk.cameras)) + " images loaded")

chunk.matchPhotos(keypoint_limit = 40000, tiepoint_limit = 10000, generic_preselection = True, reference_preselection = True)
doc.save()

chunk.alignCameras()
doc.save()

chunk.buildDepthMaps(downscale = 2, filter_mode = Metashape.MildFiltering)
doc.save()

chunk.buildModel(source_data = Metashape.DepthMapsData)
doc.save()

chunk.buildUV(page_count = 2, texture_size = 4096)
doc.save()

chunk.buildTexture(texture_size = 4096, ghosting_filter = True)
doc.save()

has_transform = chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation

if has_transform:
    chunk.buildPointCloud()
    doc.save()

    chunk.buildDem(source_data=Metashape.PointCloudData)
    doc.save()

    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
    doc.save()

# export results
chunk.exportReport(output_folder + '/report.pdf')

if chunk.model:
    chunk.exportModel(output_folder + '/model.obj')

if chunk.point_cloud:
    chunk.exportPointCloud(output_folder + '/point_cloud.las', source_data = Metashape.PointCloudData)

if chunk.elevation:
    chunk.exportRaster(output_folder + '/dem.tif', source_data = Metashape.ElevationData)

if chunk.orthomosaic:
    chunk.exportRaster(output_folder + '/orthomosaic.tif', source_data = Metashape.OrthomosaicData)

print('Processing finished, results saved to ' + output_folder + '.')
