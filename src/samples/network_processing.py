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
    print("Usage: network_processing.py <image_folder> <output_folder>")
    sys.exit(1)

image_folder = sys.argv[1]
output_folder = sys.argv[2]

network_server = '127.0.0.1'
process_network = True

Metashape.app.settings.network_path = '/mnt/datasets'

photos = find_files(image_folder, [".jpg", ".jpeg", ".tif", ".tiff"])

doc = Metashape.Document()
doc.save(output_folder + '/project.psx')

chunk = doc.addChunk()

chunk.addPhotos(photos)
doc.save()

print(str(len(chunk.cameras)) + " images loaded")

has_reference = False
for c in chunk.cameras:
    if c.reference.location:
        has_reference = True

tasks = []

task = Metashape.Tasks.MatchPhotos()
task.keypoint_limit = 40000
task.tiepoint_limit = 10000
task.generic_preselection = True
task.reference_preselection = True
tasks.append(task)

task = Metashape.Tasks.AlignCameras()
tasks.append(task)

task = Metashape.Tasks.BuildDepthMaps()
task.downscale = 2
task.filter_mode = Metashape.MildFiltering
tasks.append(task)

task = Metashape.Tasks.BuildModel()
task.source_data = Metashape.DepthMapsData
tasks.append(task)

task = Metashape.Tasks.BuildUV()
task.page_count = 2
task.texture_size = 4096
tasks.append(task)

task = Metashape.Tasks.BuildTexture()
task.texture_size = 4096
task.ghosting_filter = True
tasks.append(task)

if has_reference:
    task = Metashape.Tasks.BuildPointCloud()
    tasks.append(task)

    task = Metashape.Tasks.BuildDem()
    task.source_data = Metashape.PointCloudData
    tasks.append(task)

    task = Metashape.Tasks.BuildOrthomosaic()
    task.surface_data = Metashape.ElevationData
    tasks.append(task)

task = Metashape.Tasks.ExportReport()
task.path = output_folder + '/report.pdf'
tasks.append(task)

task = Metashape.Tasks.ExportModel()
task.path = output_folder + '/model.obj'
tasks.append(task)

task = Metashape.Tasks.ExportPointCloud()
task.path = output_folder + '/point_cloud.las'
task.source_data = Metashape.PointCloudData
tasks.append(task)

if has_reference:
    task = Metashape.Tasks.ExportRaster()
    task.path = output_folder + '/dem.tif'
    task.source_data = Metashape.ElevationData
    tasks.append(task)

    task = Metashape.Tasks.ExportRaster()
    task.path = output_folder + '/orthomosaic.tif'
    task.source_data = Metashape.OrthomosaicData
    tasks.append(task)

if process_network:
    network_tasks = []
    for task in tasks:
        if task.target == Metashape.Tasks.DocumentTarget:
            network_tasks.append(task.toNetworkTask(doc))
        else:
            network_tasks.append(task.toNetworkTask(chunk))

    client = Metashape.NetworkClient()
    client.connect(network_server)
    batch_id = client.createBatch(doc.path, network_tasks)
    client.setBatchPaused(batch_id, False)

    print('Processing started, results will be saved to ' + output_folder + '.')
else:
    for task in tasks:
        if task.target == Metashape.Tasks.DocumentTarget:
            task.apply(doc)
        else:
            task.apply(chunk)

    print('Processing finished, results saved to ' + output_folder + '.')
