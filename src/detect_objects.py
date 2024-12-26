# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts
#
# See https://agisoft.freshdesk.com/support/solutions/articles/31000162552-automatic-detection-of-objects-on-orthomosaic
# Based on https://github.com/weecology/DeepForest (tested on deepforest==1.0.8)
#
# This is a neural network assistant for objects (trees/cars/sea lions/etc.) detection on orthomosaic.
# This script can help you to detect trees or other objects (like cars or sea lions)
# using partial manual annotations to guide the neural network what to seek for.
# The pre-trained network was pre-trained for tree detection task, but the results are much better
# if you will annotate small region to guide the neural network (it will be trained additionally).
# Note that you need NVIDIA GPU for fast processing (i.e. CUDA-compatible GPU required, CPU is supported too but it is very slow)
#
# How to install (Linux):
#
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
#    copy detect_objects.py script to /home/<username>/.local/share/Agisoft/Metashape Pro/scripts/
# 2. Restart Metashape.
#
# How to install (Windows):
#
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
#    copy detect_objects.py script to C:/Users/<username>/AppData/Local/Agisoft/Metashape Pro/scripts/
# 2. Restart Metashape
#
# How to use:
#
# 1. Open a dataset with an orthomosaic with at least 10 cm/pix resolution (i.e. GSD should be <= 10 cm/pix), 10 cm/pix or 5 cm/pix are recommended. Note that path should not include non-ascii characters.
# 2. Create a shape layer 'Train zones' with at least one axis aligned bounding box (using 'Draw Rectangle') specifing the training zone
#    (each of its sides should be around 50-60 meters), it is recommended to specify color for shapes of this layer - red for example
# 3. Create a shape layer 'Train data' with all trees (or all cars) in train zones specified as axis aligned bounding box (using 'Draw Rectangle'),
#    it is recommended to specify different color for shapes of this layer - blue for example
# 4. Ensure that you didn't miss any objects (trees if you want to detect trees, or cars if you want to detect cars) in train zones
# 5. Press 'Custom menu/Detect objects'
# 6. Ensure that proper shape layers are selected as Train zones and Train data
# 7. Press Run
#
# To process detection only on some part of the orthomosaic please specify Outer Boundary
# (ensure that Train zones are inside the Outer Boundary):
# https://www.agisoft.com/forum/index.php?topic=4910.msg24580#msg24580
#
# How to use a pre-trained neural network model for tree detection:
# (not recommended because results are much better after training on annotated 50x50 meters zone)
#
# 1. Open a dataset with a orthomosaic with at least 10 cm/pix resolution (i.e. GSD should be <= 10 cm/pix)
# 2. Press 'Custom menu/Detect objects'
# 3. Press Run
#
# If you will encounter error like this:
#      Downloading: "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth" to C:\Users\<username>/.cache\torch\hub\checkpoints\retinanet_resnet50_fpn_coco-eeacb38b.pth
#      Traceback (most recent call last):
#        File "C:\Program Files\Agisoft\Metashape Pro\python\lib\urllib\request.py", line 1354, in do_open
#          h.request(req.get_method(), req.selector, req.data, headers,
#        ...
#        File "C:\Program Files\Agisoft\Metashape Pro\python\lib\socket.py", line 918, in getaddrinfo
#          for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
#      socket.gaierror: [Errno 11002] getaddrinfo failed
#      During handling of the above exception, another exception occurred:
#        ...
#    Then you need to manually download file at URL from the begining of the error message (similar to https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth)
#    to the specified in the same line directory (C:\Users\<username>\.cache\torch\hub\checkpoints\)


import Metashape
import pathlib, shutil, os, time
from PySide2 import QtGui, QtCore, QtWidgets

import urllib.request
from modules.pip_auto_install import pip_install, user_packages_location, _is_already_installed

# Checking compatibility
compatible_major_version = "2.2"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

pathlib.Path(user_packages_location).mkdir(parents=True, exist_ok=True)
temporary_file = os.path.join(user_packages_location, "temp_links.html")

requirements_txt = """-f "{find_links_file_path}"
-f https://download.pytorch.org/whl/torch_stable.html
albumentations==1.0.3
deepforest==1.2.4
pytorch-lightning==1.5.10
torch==1.9.0+cu111
torchvision==0.10.0+cu111
torchaudio===0.9.0

absl-py==2.1.0
affine==2.4.0
aiohttp==3.9.3
aiosignal==1.3.1
async-timeout==4.0.3
attrs==23.2.0
certifi==2024.2.2
click==8.1.7
click-plugins==1.1.1
cligj==0.7.2
colorama==0.4.6
contourpy==1.2.0
cycler==0.12.1
fiona==1.9.6
fonttools==4.50.0
frozenlist==1.4.1
fsspec==2024.3.1
future==1.0.0
geopandas==0.14.3
grpcio==1.62.1
idna==3.6
imagecodecs==2024.1.1
imageio==2.34.0
importlib_metadata==7.1.0
importlib_resources==6.3.2
kiwisolver==1.4.5
lazy_loader==0.3
lightning-utilities==0.11.0
Markdown==3.6
MarkupSafe==2.1.5
matplotlib==3.8.3
multidict==6.0.5
networkx==3.2.1
numpy==1.26.4
opencv-python==4.9.0.80
opencv-python-headless==4.9.0.80
packaging==24.0
pandas==2.2.1
pillow==10.2.0
progressbar2==4.4.2
protobuf==5.26.0
psutil==5.9.8
pyDeprecate==0.3.1
pyparsing==3.1.2
pyproj==3.6.1
python-dateutil==2.9.0.post0
python-utils==3.8.2
pytz==2024.1
PyYAML==6.0.1
rasterio==1.3.9
Rtree==1.2.0
scikit-image==0.22.0
scipy==1.12.0
shapely==2.0.3
six==1.16.0
slidingwindow==0.0.14
snuggs==1.4.7
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tifffile==2024.2.12
torchmetrics==1.2.1
tqdm==4.66.2
typing_extensions==4.10.0
tzdata==2024.1
Werkzeug==3.0.1
xmltodict==0.13.0
yarl==1.9.4
zipp==3.18.1""".format(find_links_file_path=temporary_file)

# Avoid network request if requirements already installed
if not _is_already_installed(requirements_txt):
    find_links_file_url = "https://raw.githubusercontent.com/agisoft-llc/metashape-scripts/master/misc/links.html"
    urllib.request.urlretrieve(find_links_file_url, temporary_file)
    pip_install(requirements_txt)

def pandas_append(df, row, ignore_index=False):
    import pandas as pd
    if isinstance(row, pd.DataFrame):
        result = pd.concat([df, row], ignore_index=ignore_index)
    elif isinstance(row, pd.core.series.Series):
        result = pd.concat([df, row.to_frame().T], ignore_index=ignore_index)
    elif isinstance(row, dict):
        result = pd.concat([df, pd.DataFrame(row, index=[0], columns=df.columns)])
    else:
        raise RuntimeError("pandas_append: unsupported row type - {}".format(type(row)))
    return result

def getShapeVertices(shape):
    chunk = Metashape.app.document.chunk
    if (chunk == None):
        raise Exception("Null chunk")

    T = chunk.transform.matrix
    result = []

    if shape.is_attached:
        assert(len(shape.geometry.coordinates) == 1)
        for key in shape.geometry.coordinates[0]:
            for marker in chunk.markers:
                if marker.key == key:
                    if (not marker.position):
                        raise Exception("Invalid shape vertex")

                    point = T.mulp(marker.position)
                    point = Metashape.CoordinateSystem.transform(point, chunk.world_crs, chunk.shapes.crs)
                    result.append(point)
    else:
        assert(len(shape.geometry.coordinates) == 1)
        for coord in shape.geometry.coordinates[0]:
            result.append(coord)

    return result

class DetectObjectsDlg(QtWidgets.QDialog):

    def __init__(self, parent):

        self.force_small_patch_size = False
        # Set force_small_patch_size to True if you want to train on small zones with a lot of small objects (train zone still should be at least 400*orthomosaic_resolution)
        # For example, with force_small_patch_size=True and orthomosaic resolution=5cm - train zone should be >= 20x20 m. If orthomosaic=2.5cm - train zone should be >= 10x10 m.
        # Note that this can work only if:
        # 1) You have very small objects
        # 2) Train zone contains a lot of objects

        self.augment_colors = True
        # Set augment_colors to False if you have bad results and you want to force neural network to take into account color of objects

        self.expected_layer_name_train_zones = "Train zone"
        self.expected_layer_name_train_data  = "Train data"
        self.layer_name_detection_data       = "Detected data"

        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent / "objects_detection")
        else:
            self.working_dir = ""

        self.save_model_path = ""
        self.load_model_path = self.readModelLoadPathFromSettings()

        self.cleanup_working_dir = False
        self.debug_tiles = False

        self.train_on_user_data_enabled = False

        self.max_epochs = 20  # bigger number of epochs leads to better neural network training (but slower)
        self.data_augmentation_multiplier = 8  # from 1 to 8, bigger multiplier leads to better neural network training (but slower)
        self.preferred_patch_size = 400  # 400 pixels
        self.preferred_resolution = 0.10  # 10 cm/pix
        self.detection_score_threshold = None  # can be from 0.0 to 1.0, for example it can be 0.98

        self.prefer_original_resolution = True
        self.use_neural_network_pretrained_on_birds = False

        self.tiles_without_annotations_supported = False  # See https://github.com/weecology/DeepForest/issues/216

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Objects detection on orthomosaic")

        self.chunk = Metashape.app.document.chunk
        self.create_gui()

        self.exec()

    def stop(self):
        self.stopped = True

    def check_stopped(self):
        if self.stopped:
            raise InterruptedError("Stop was pressed")

    def process(self):
        try:
            self.stopped = False
            self.btnRun.setEnabled(False)
            self.btnStop.setEnabled(True)

            time_start = time.time()

            self.load_params()

            self.prepair()

            print("Script started...")

            self.create_neural_network()

            self.export_orthomosaic()

            if self.chunk.shapes is None:
                self.chunk.shapes = Metashape.Shapes()
                self.chunk.shapes.crs = self.chunk.crs

            if self.train_on_user_data_enabled:
                self.train_on_user_data()

            if len(self.save_model_path) > 0:
                self.saveToSettingsModelLoadPath(self.save_model_path)
            else:
                self.saveToSettingsModelLoadPath(self.load_model_path)

            self.detect()

            self.results_time_total = time.time() - time_start

            self.show_results_dialog()
        except:
            if self.stopped:
                Metashape.app.messageBox("Processing was stopped.")
            else:
                Metashape.app.messageBox("Something gone wrong.\n"
                                         "Please check the console.")
                raise
        finally:
            if self.cleanup_working_dir:
                shutil.rmtree(self.working_dir, ignore_errors=True)
            self.reject()

        print("Script finished.")
        return True

    def prepair(self):
        import os, sys, multiprocessing
        import random, string

        if self.working_dir == "":
            raise Exception("You should specify working directory (or save .psx project)")

        print("Working dir: {}".format(self.working_dir))
        try:
            os.mkdir(self.working_dir)
        except FileExistsError:
            already_existing_working_dir = self.working_dir
            random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            self.working_dir = self.working_dir + "/tmp_" + random_suffix
            print("Working dir: {} already exists, trying instead: {}".format(already_existing_working_dir, self.working_dir))
            try:
                os.mkdir(self.working_dir)
            except FileExistsError:
                raise Exception("Working directory {} already exists! Please specify another working dir.".format(self.working_dir))

        self.cleanup_working_dir = True

        self.dir_tiles = self.working_dir + "/tiles/"

        self.dir_train_data = self.working_dir + "/train/"
        self.dir_train_subtiles = self.dir_train_data + "inner/"
        self.dir_train_subtiles_debug = self.dir_train_subtiles + "debug/"

        self.dir_detection_results = self.working_dir + "/detection/"
        self.dir_subtiles_results = self.dir_detection_results + "inner/"

        for subdir in [self.dir_tiles, self.dir_train_data, self.dir_train_subtiles, self.dir_train_subtiles_debug, self.dir_detection_results, self.dir_subtiles_results]:
            shutil.rmtree(subdir, ignore_errors=True)
            os.mkdir(subdir)

        import torch
        torch_hub_dir = torch.hub.get_dir()

        from deepforest import utilities
        if not hasattr(utilities, '__models_dir_path_already_patched__'):
            original_use_release = utilities.use_release
            original_use_bird_release = utilities.use_bird_release
            # This is a workaround for Windows permission issues (we can't easily download files into .../site-packages/deepforest/...)
            def patched_use_release(**kwargs):
                kwargs["save_dir"] = torch_hub_dir
                return original_use_release(**kwargs)
            def patched_use_bird_release(**kwargs):
                kwargs["save_dir"] = torch_hub_dir
                return original_use_bird_release(**kwargs)
            utilities.use_release = patched_use_release
            utilities.use_bird_release = patched_use_bird_release
            utilities.__models_dir_path_already_patched__ = True

        if os.name == 'nt': # if Windows
            multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'python.exe'))

    def create_neural_network(self):
        print("Neural network loading...")
        import torch
        import deepforest
        import deepforest.main

        self.m = deepforest.main.deepforest()

        if len(self.load_model_path) > 0:
            self.m.use_release()
            print("Using the neural network loaded from '{}'...".format(self.load_model_path))
            self.m.model = torch.load(self.load_model_path)
        else:
            if self.use_neural_network_pretrained_on_birds:
                # use neural network pre-trained on birds
                print("Using the neural network pre-trained on birds...")
                self.m.use_bird_release()
            else:
                # use neural network pre-trained on trees
                print("Using the neural network pre-trained on trees...")
                self.m.use_release()

    def export_orthomosaic(self):
        import numpy as np

        print("Prepairing orthomosaic...")

        kwargs = {}
        if not self.prefer_original_resolution and (self.chunk.orthomosaic.resolution < self.preferred_resolution*0.90):
            kwargs["resolution"] = self.preferred_resolution
        else:
            print("no resolution downscaling required")
        self.chunk.exportRaster(path=self.dir_tiles + "tile.jpg", source_data=Metashape.OrthomosaicData, image_format=Metashape.ImageFormat.ImageFormatJPEG, save_alpha=False, white_background=True,
                                save_world=True,
                                split_in_blocks=True, block_width=self.patch_size, block_height=self.patch_size,
                                **kwargs)

        tiles = os.listdir(self.dir_tiles)
        self.tiles_paths = {}
        self.tiles_to_world = {}
        for tile in sorted(tiles):
            assert tile.startswith("tile-")

            _, tile_x, tile_y = tile.split(".")[0].split("-")
            tile_x, tile_y = map(int, [tile_x, tile_y])
            if tile.endswith(".jgw") or tile.endswith(".pgw"):  # https://en.wikipedia.org/wiki/World_file
                with open(self.dir_tiles + tile, "r") as file:
                    matrix2x3 = list(map(float, file.readlines()))
                matrix2x3 = np.array(matrix2x3).reshape(3, 2).T
                self.tiles_to_world[tile_x, tile_y] = matrix2x3
            elif tile.endswith(".jpg"):
                self.tiles_paths[tile_x, tile_y] = self.dir_tiles + tile

        assert(len(self.tiles_paths) == len(self.tiles_to_world))
        assert(self.tiles_paths.keys() == self.tiles_to_world.keys())

        self.tile_min_x = min([key[0] for key in self.tiles_paths.keys()])
        self.tile_max_x = max([key[0] for key in self.tiles_paths.keys()])
        self.tile_min_y = min([key[1] for key in self.tiles_paths.keys()])
        self.tile_max_y = max([key[1] for key in self.tiles_paths.keys()])
        print("{} tiles, tile_x in [{}; {}], tile_y in [{}; {}]".format(len(self.tiles_paths), self.tile_min_x, self.tile_max_x, self.tile_min_y, self.tile_max_y))

    def train_on_user_data(self):
        import sys
        import cv2
        import random
        import numpy as np
        import pandas as pd
        import multiprocessing
        import torch
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Callback

        random.seed(2391231231324531)

        app = QtWidgets.QApplication.instance()

        training_start = time.time()
        print("Neural network additional training on user data...")

        self.train_zones_on_ortho = []

        n_train_zone_shapes_out_of_orthomosaic = 0
        for zone_i, shape in enumerate(self.train_zones):
            shape_vertices = getShapeVertices(shape)
            zone_from_world = None
            zone_from_world_best = None
            for tile_x in range(self.tile_min_x, self.tile_max_x + 1):
                for tile_y in range(self.tile_min_y, self.tile_max_y + 1):
                    if (tile_x, tile_y) not in self.tiles_paths:
                        continue
                    to_world = self.tiles_to_world[tile_x, tile_y]
                    from_world = self.invert_matrix_2x3(to_world)
                    for p in shape_vertices:
                        p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                        p_in_tile = from_world @ [p.x, p.y, 1]
                        distance2_to_tile_center = np.linalg.norm(p_in_tile - [self.patch_size/2, self.patch_size/2])
                        if zone_from_world_best is None or distance2_to_tile_center < zone_from_world_best:
                            zone_from_world_best = distance2_to_tile_center
                            zone_from_world = self.invert_matrix_2x3(self.add_pixel_shift(to_world, -tile_x * self.patch_size, -tile_y * self.patch_size))
            if zone_from_world_best > 1.1 * (self.patch_size / 2)**2:
                n_train_zone_shapes_out_of_orthomosaic += 1

            zone_from = None
            zone_to = None
            for p in shape_vertices:
                p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                p_in_ortho = np.int32(np.round(zone_from_world @ [p.x, p.y, 1]))
                if zone_from is None:
                    zone_from = p_in_ortho
                if zone_to is None:
                    zone_to = p_in_ortho
                zone_from = np.minimum(zone_from, p_in_ortho)
                zone_to = np.maximum(zone_to, p_in_ortho)
            train_size = zone_to - zone_from
            train_size_m = np.int32(np.round(train_size * self.orthomosaic_resolution))
            if np.any(train_size < self.patch_size):
                print("Train zone #{} {}x{} pixels ({}x{} meters) is too small - each side should be at least {} meters"
                      .format(zone_i + 1, train_size[0], train_size[1], train_size_m[0], train_size_m[1], self.patch_size * self.orthomosaic_resolution), file=sys.stderr)
                self.train_zones_on_ortho.append(None)
            else:
                print("Train zone #{}: {}x{} orthomosaic pixels, {}x{} meters".format(zone_i + 1, train_size[0], train_size[1], train_size_m[0], train_size_m[1]))
                self.train_zones_on_ortho.append((zone_from, zone_to, zone_from_world))
        assert len(self.train_zones_on_ortho) == len(self.train_zones)

        if n_train_zone_shapes_out_of_orthomosaic > 0:
            print("Warning, {} train zones shapes are out of orthomosaic".format(n_train_zone_shapes_out_of_orthomosaic))

        area_threshold = 0.3

        all_annotations = pd.DataFrame(columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
        nannotated_tiles = 0

        if self.tiles_without_annotations_supported:
            empty_tile_name = "empty_tile.jpg"
            empty_tile = self.create_empty_tile()
            cv2.imwrite(self.dir_train_subtiles + empty_tile_name, empty_tile)

            # See https://github.com/weecology/DeepForest/issues/216
            all_annotations = pandas_append(all_annotations, {'image_path': empty_tile_name, 'xmin': '0', 'ymin': '0', 'xmax': '0', 'ymax': '0', 'label': 'Tree'}, ignore_index=True)

        nempty_tiles = 0

        self.train_nannotations_in_zones = 0
        for zone_i, shape in enumerate(self.train_zones):
            if self.train_zones_on_ortho[zone_i] is None:
                continue
            zone_from, zone_to, zone_from_world = self.train_zones_on_ortho[zone_i]
            annotations = []
            for annotation in self.train_data:
                annotation_vertices = getShapeVertices(annotation)
                annotation_from = None
                annotation_to = None
                for p in annotation_vertices:
                    p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                    p_in_ortho = np.int32(np.round(zone_from_world @ [p.x, p.y, 1]))
                    if annotation_from is None:
                        annotation_from = p_in_ortho
                    if annotation_to is None:
                        annotation_to = p_in_ortho
                    annotation_from = np.minimum(annotation_from, p_in_ortho)
                    annotation_to = np.maximum(annotation_to, p_in_ortho)
                bbox_from, bbox_to = self.intersect(zone_from, zone_to, annotation_from, annotation_to)
                if self.area(bbox_from, bbox_to) > self.area(annotation_from, annotation_to) * area_threshold:
                    annotations.append((annotation_from, annotation_to))
            self.train_nannotations_in_zones += len(annotations)
            print("Train zone #{}: {} annotations inside".format(zone_i + 1, len(annotations)))

            border = self.patch_inner_border
            inner_path_size = self.patch_size - 2 * border

            zone_size = zone_to - zone_from
            assert np.all(zone_size >= self.patch_size)
            nx_tiles, ny_tiles = np.int32((zone_size - 2 * border + inner_path_size - 1) // inner_path_size)
            assert nx_tiles >= 1 and ny_tiles >= 1
            xy_step = np.int32(np.round((zone_size + [nx_tiles, ny_tiles] - 1) // [nx_tiles, ny_tiles]))
            out_of_orthomosaic_train_tile = 0
            for x_tile in range(0, nx_tiles):
                for y_tile in range(0, ny_tiles):
                    tile_to = zone_from + self.patch_size + xy_step * [x_tile, y_tile]
                    if x_tile == nx_tiles - 1 and y_tile == ny_tiles - 1:
                        assert np.all(tile_to >= zone_to)
                    tile_to = np.minimum(tile_to, zone_to)
                    tile_from = tile_to - self.patch_size
                    if x_tile == 0 and y_tile == 0:
                        assert np.all(tile_from == zone_from)
                    assert np.all(tile_from >= zone_from)

                    tile = self.read_part(tile_from, tile_to)
                    assert tile.shape == (self.patch_size, self.patch_size, 3)

                    if np.all(tile == 255):
                        out_of_orthomosaic_train_tile += 1
                        continue

                    tile_annotations = []
                    for annotation_from, annotation_to in annotations:
                        bbox_from, bbox_to = self.intersect(tile_from, tile_to, annotation_from, annotation_to)
                        if self.area(bbox_from, bbox_to) > self.area(annotation_from, annotation_to) * area_threshold:
                            tile_annotations.append((bbox_from - tile_from, bbox_to - tile_from))

                    max_augmented_versions = 8
                    all_augmented_versions = list(range(max_augmented_versions))

                    augmented_versions = []
                    augmented_versions_to_add = max(1, self.data_augmentation_multiplier)
                    while augmented_versions_to_add > 0:
                        if augmented_versions_to_add < max_augmented_versions:
                            shuffled_augmented_versions = all_augmented_versions
                            random.shuffle(shuffled_augmented_versions)
                            augmented_versions.extend(shuffled_augmented_versions[:augmented_versions_to_add])
                            augmented_versions_to_add = 0
                        else:
                            augmented_versions.extend(all_augmented_versions)
                            augmented_versions_to_add -= max_augmented_versions

                    for version_i in augmented_versions:
                        tile_version = tile
                        tile_annotations_version = tile_annotations

                        is_mirrored = ((version_i % 4) == 1)
                        n90rotation = (version_i % 4)
                        if is_mirrored:
                            tile_annotations_version = self.flip_annotations(tile_annotations_version, tile_version)
                            tile_version = cv2.flip(tile_version, 0)
                        for rotation_i in range(n90rotation):
                            tile_annotations_version = self.rotate90clockwise_annotations(tile_annotations_version, tile_version)
                            tile_version = cv2.rotate(tile_version, cv2.ROTATE_90_CLOCKWISE)

                        tile_version = self.random_augmentation(tile_version)

                        tile_name = "{}-{}-{}-{}.jpg".format((zone_i + 1), x_tile, y_tile, version_i)

                        nannotated_tiles += 1
                        for (xmin, ymin), (xmax, ymax) in tile_annotations_version:
                            all_annotations = pandas_append(all_annotations, {'image_path': tile_name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'label': 'Tree'}, ignore_index=True)
                        if len(tile_annotations_version) == 0:
                            if self.tiles_without_annotations_supported:
                                all_annotations = pandas_append(all_annotations, {'image_path': tile_name, 'xmin': '0', 'ymin': '0', 'xmax': '0', 'ymax': '0', 'label': 'Tree'}, ignore_index=True)
                            nempty_tiles += 1

                        cv2.imwrite(self.dir_train_subtiles + tile_name, tile_version)
                        if self.debug_tiles:
                            tile_with_trees = self.debug_draw_trees(tile_version, tile_annotations_version)
                            cv2.imwrite(self.dir_train_subtiles_debug + tile_name, tile_with_trees)

            if out_of_orthomosaic_train_tile == nx_tiles * ny_tiles:
                raise RuntimeError("It seems that zone #{} has no orthomosaic data, please check zones, orthomosaic and its Outer Boundary.".format(zone_i + 1))
            else:
                if out_of_orthomosaic_train_tile > 0:
                    print("{}/{} of tiles in zone #{} has no orthomosaic data".format(out_of_orthomosaic_train_tile, nx_tiles * ny_tiles, zone_i + 1))

        print("{} tiles ({} empty{}) for training prepared with {} annotations"
              .format(nannotated_tiles, nempty_tiles, " - they are not supported" if (nempty_tiles > 0 and not self.tiles_without_annotations_supported) else "", len(all_annotations)))
        print("Training with {} epochs and x{} augmentations (augment colors: {})...".format(self.max_epochs, self.data_augmentation_multiplier, self.augment_colors))
        self.freeze_layers()

        annotations_file = self.dir_train_subtiles + "annotations.csv"
        all_annotations.to_csv(annotations_file, header=True, index=False)

        class MyCallback(Callback):
            def __init__(self, thiz_dlg):
                self.nepochs_done = 0
                self.nepochs = thiz_dlg.max_epochs
                self.pbar = thiz_dlg.trainPBar
                self.thiz_dlg = thiz_dlg
            def on_epoch_end(self, trainer, pl_module):
                self.nepochs_done += 1
                self.pbar.setValue(self.nepochs_done * 100 / self.nepochs)
                Metashape.app.update()
                app.processEvents()
                self.thiz_dlg.check_stopped()

        if torch.cuda.device_count() > 0:
            print("Using GPU...")
            trainer_gpus = 1
            trainer_auto_select_gpus = True
        else:
            print("Using CPU (will be very slow)...")
            trainer_gpus = 0
            trainer_auto_select_gpus = False
            torch.set_num_threads(multiprocessing.cpu_count())

        trainer = Trainer(max_epochs=self.max_epochs, gpus=trainer_gpus, auto_select_gpus=trainer_auto_select_gpus, callbacks=[MyCallback(self)], checkpoint_callback=False, logger=False)
        train_ds = self.m.load_dataset(annotations_file, root_dir=os.path.dirname(annotations_file))
        trainer.fit(self.m, train_ds)

        self.results_time_training = time.time() - training_start

        if len(self.save_model_path) > 0:
            torch.save(self.m.model, self.save_model_path)
            print("Model trained on {} annotations with {} m/pix resolution saved to '{}'".format(self.train_nannotations_in_zones, self.orthomosaic_resolution, self.save_model_path))

    def create_empty_tile(self):
        import numpy as np

        empty_tile = np.zeros((self.patch_size, self.patch_size, 3), np.uint8)
        empty_tile[:, :, :] = 255
        return empty_tile

    def freeze_all_layers(self):
        for p in self.m.model.backbone.parameters():
            p.requires_grad = False

    def freeze_low_layers(self, freezeConv1=False, freezeUpToLevel=2):
        body = self.m.model.backbone.body
        layers = [body.layer1, body.layer2, body.layer3, body.layer4]
        assert (0 <= freezeUpToLevel <= len(layers))

        if freezeConv1:
            for p in self.m.model.backbone.body.conv1.parameters():
                p.requires_grad = False

        for layer in layers[:freezeUpToLevel]:
            for p in layer.parameters():
                p.requires_grad = False

    def freeze_layers(self):
        self.freeze_low_layers(freezeConv1=False, freezeUpToLevel=2)

    def flip_annotations(self, trees, img):
        # x, y -> x, h-y
        import numpy as np

        h, w, cn = img.shape
        flipped_trees = []
        for bbox_from, bbox_to in trees:
            assert np.all(bbox_from >= np.int32([0, 0]))
            assert np.all(bbox_to <= np.int32([w, h]))
            (xmin, ymin), (xmax, ymax) = bbox_from, bbox_to
            flipped_trees.append(((xmin, h - ymax), (xmax, h - ymin)))
        return flipped_trees

    def rotate90clockwise_point(self, x, y, w, h):
        return h-y, x

    def rotate90clockwise_annotations(self, trees, img):
        # 0, 0 -> w, 0
        # x, y -> h-y, x
        import numpy as np

        h, w, cn = img.shape
        rotated_trees = []
        for bbox_from, bbox_to in trees:
            assert np.all(bbox_from >= np.int32([0, 0]))
            assert np.all(bbox_to <= np.int32([w, h]))
            (xmin, ymin), (xmax, ymax) = bbox_from, bbox_to
            xmin2, ymin2 = self.rotate90clockwise_point(xmin, ymin, w, h)
            xmax2, ymax2 = self.rotate90clockwise_point(xmax, ymax, w, h)
            xmin2, xmax2 = min(xmin2, xmax2), max(xmin2, xmax2)
            ymin2, ymax2 = min(ymin2, ymax2), max(ymin2, ymax2)
            rotated_trees.append(((xmin2, ymin2), (xmax2, ymax2)))
        return rotated_trees

    def random_augmentation(self, img):
        import cv2
        import albumentations as A

        stages = []
        if self.augment_colors:
            stages.append(A.HueSaturationValue(hue_shift_limit=360, sat_shift_limit=30, val_shift_limit=20, always_apply=True))
        stages.append(A.ISONoise(p=0.5))
        if self.augment_colors:
            stages.append(A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, p=0.5))

        transform = A.Compose(stages)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(image=img)["image"]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def read_part(self, res_from, res_to):
        import cv2
        import numpy as np

        res_size = res_to - res_from
        assert np.all(res_size >= [self.patch_size, self.patch_size])
        res = np.zeros((res_size[1], res_size[0], 3), np.uint8)
        res[:, :, :] = 255

        tile_xy_from = np.int32(res_from // self.patch_size)
        tile_xy_upto = np.int32((res_to - 1) // self.patch_size)
        assert np.all(tile_xy_from <= tile_xy_upto)
        for tile_x in range(tile_xy_from[0], tile_xy_upto[0] + 1):
            for tile_y in range(tile_xy_from[1], tile_xy_upto[1] + 1):
                if (tile_x, tile_y) not in self.tiles_paths:
                    continue
                part = cv2.imread(self.tiles_paths[tile_x, tile_y])
                part = cv2.copyMakeBorder(part, 0, self.patch_size - part.shape[0], 0, self.patch_size - part.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
                part_from = np.int32([tile_x, tile_y]) * self.patch_size - res_from
                part_to = part_from + self.patch_size

                res_inner_from = np.int32([max(0, part_from[0]), max(0, part_from[1])])
                res_inner_to = np.int32([min(part_to[0], res_size[0]), min(part_to[1], res_size[1])])

                part_inner_from = res_inner_from - part_from
                part_inner_to = part_inner_from + res_inner_to - res_inner_from

                res[res_inner_from[1]:res_inner_to[1], res_inner_from[0]:res_inner_to[0], :] = part[part_inner_from[1]:part_inner_to[1], part_inner_from[0]:part_inner_to[0], :]

        return res

    def intersect(self, a_from, a_to, b_from, b_to):
        import numpy as np
        c_from = np.maximum(a_from, b_from)
        c_to = np.minimum(a_to, b_to)
        if np.any(c_from >= c_to):
            return c_from, c_from
        else:
            return c_from, c_to

    def area(self, a_from, a_to):
        a_size = a_to - a_from
        return a_size[0] * a_size[1]

    def add_pixel_shift(self, to_world, dx, dy):
        to_world = to_world.copy()
        to_world[0, 2] = to_world[0, :] @ [dx, dy, 1]
        to_world[1, 2] = to_world[1, :] @ [dx, dy, 1]
        return to_world

    def invert_matrix_2x3(self, to_world):
        import numpy as np

        to_world33 = np.vstack([to_world, [0, 0, 1]])
        from_world = np.linalg.inv(to_world33)

        assert(from_world[2, 0] == from_world[2, 1] == 0)
        assert(from_world[2, 2] == 1)
        from_world = from_world[:2, :]

        return from_world

    def detect(self):
        import cv2
        import numpy as np
        import pandas as pd

        app = QtWidgets.QApplication.instance()

        print("Detection...")
        time_start = time.time()

        detected_label = self.layer_name_detection_data + " ({:.2f} cm/pix".format(100.0 * self.orthomosaic_resolution)
        if self.use_neural_network_pretrained_on_birds:
            detected_label += ", birds pre-train"
        if self.augment_colors:
            detected_label += ", any color"
        if self.train_on_user_data_enabled:
            detected_label += ", trained on {} shapes in {} zones".format(self.train_nannotations_in_zones, len(self.train_zones))
        detected_label += ")"

        detected_shapes_layer = self.chunk.shapes.addGroup()
        detected_shapes_layer.label = detected_label

        ntrees_detected = 0

        big_tiles_k = 8
        border = self.patch_inner_border
        area_overlap_threshold = 0.60

        big_tiles = set()
        for tile_x in range(self.tile_min_x, self.tile_max_x + 1):
            for tile_y in range(self.tile_min_y, self.tile_max_y + 1):
                if (tile_x, tile_y) not in self.tiles_paths:
                    continue
                big_tile_x, big_tile_y = tile_x // big_tiles_k, tile_y // big_tiles_k
                big_tiles.add((big_tile_x, big_tile_y))

        bigtiles_trees = {}
        bigtiles_to_world = {}
        bigtiles_idx_on_borders = {}

        for big_tile_index, (big_tile_x, big_tile_y) in enumerate(sorted(big_tiles)):
            big_tile = np.zeros((border + big_tiles_k*self.patch_size + border, border + big_tiles_k*self.patch_size + border, 3), np.uint8)
            big_tile[:, :, :] = 255
            big_tile_to_world = None

            for xi in range(-1, big_tiles_k + 1):
                for yi in range(-1, big_tiles_k + 1):
                    tile_x, tile_y = big_tiles_k * big_tile_x + xi, big_tiles_k * big_tile_y + yi
                    if (tile_x, tile_y) not in self.tiles_paths:
                        continue
                    part = cv2.imread(self.tiles_paths[tile_x, tile_y])
                    part = cv2.copyMakeBorder(part, 0, self.patch_size - part.shape[0], 0, self.patch_size - part.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    if xi in [-1, big_tiles_k] or yi in [-1, big_tiles_k]:
                        fromx, fromy = border + xi * self.patch_size, border + yi * self.patch_size
                        tox, toy = fromx + self.patch_size, fromy + self.patch_size
                        if xi == -1:
                            part = part[:, self.patch_size - border:, :]
                            fromx += self.patch_size - border
                        if xi == big_tiles_k:
                            part = part[:, :border, :]
                            tox = fromx + border
                        if yi == -1:
                            part = part[self.patch_size - border:, :, :]
                            fromy += self.patch_size - border
                        if yi == big_tiles_k:
                            part = part[:border, :, :]
                            toy = fromy + border
                        big_tile[fromy:toy, fromx:tox, :] = part
                    else:
                        big_tile[border + yi * self.patch_size:, border + xi * self.patch_size:, :][:self.patch_size, :self.patch_size, :] = part
                        big_tile_to_world = self.add_pixel_shift(self.tiles_to_world[tile_x, tile_y], -(border + xi * self.patch_size), -(border + yi * self.patch_size))

            assert big_tile_to_world is not None

            subtiles_trees = {}
            tile_inner_size = self.patch_size - 2*border
            inner_tiles_nx = (big_tile.shape[1]-2*border+tile_inner_size-1)//tile_inner_size
            inner_tiles_ny = (big_tile.shape[0]-2*border+tile_inner_size-1)//tile_inner_size
            for xi in range(inner_tiles_nx):
                for yi in range(inner_tiles_ny):
                    tox, toy = min(big_tile.shape[1], 2*border+(xi + 1) * tile_inner_size), min(big_tile.shape[0], 2*border+(yi + 1) * tile_inner_size)
                    fromx, fromy = tox - self.patch_size, toy - self.patch_size
                    subtile = big_tile[fromy:toy, fromx:tox, :]

                    white_pixels_fraction = np.sum(np.all(subtile == 255, axis=-1)) / (subtile.shape[0] * subtile.shape[1])

                    assert(subtile.shape == (self.patch_size, self.patch_size, 3))
                    subtile_trees = self.m.predict_image(subtile.astype('float32'))
                    Metashape.app.update()
                    app.processEvents()
                    self.check_stopped()
                    if subtile_trees is not None:
                        subtile_inner_trees = pd.DataFrame(columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
                        subtile_inner_trees_debug = pd.DataFrame(columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
                        for idx, row in subtile_trees.iterrows():
                            xmin, ymin, xmax, ymax, label, score = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax), row.label, row.score
                            assert (label == "Tree")
                            if xmin >= self.patch_size - border or xmax <= border or ymin >= self.patch_size - border or ymax <= border:
                                continue
                            if self.detection_score_threshold is not None and score < self.detection_score_threshold:
                                continue
                            if white_pixels_fraction > 0.10:
                                subtile_bbox = subtile[ymin:ymax, xmin:xmax, :]
                                bbox_white_pixels_fraction = np.sum(np.all(subtile_bbox == 255, axis=-1)) / (subtile_bbox.shape[0] * subtile_bbox.shape[1])
                                if bbox_white_pixels_fraction > 0.70:
                                    continue
                            xmin, xmax = map(lambda x: fromx + x, [xmin, xmax])
                            ymin, ymax = map(lambda y: fromy + y, [ymin, ymax])
                            subtile_inner_trees_debug = pandas_append(subtile_inner_trees_debug, row, ignore_index=True)
                            row.xmin, row.ymin, row.xmax, row.ymax = xmin, ymin, xmax, ymax
                            subtile_inner_trees = pandas_append(subtile_inner_trees, row, ignore_index=True)

                        if self.debug_tiles:
                            img_with_trees = self.debug_draw_trees(subtile, subtile_trees)
                            cv2.imwrite(self.dir_subtiles_results + "{}-{}-{}-{}.jpg".format(big_tile_x, big_tile_y, xi, yi), img_with_trees)
                            img_with_inner_trees = self.debug_draw_trees(subtile, subtile_inner_trees_debug)
                            cv2.imwrite(self.dir_subtiles_results + "{}-{}-{}-{}_inner.jpg".format(big_tile_x, big_tile_y, xi, yi), img_with_inner_trees)
                    else:
                        subtile_inner_trees = pd.DataFrame(columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
                        if self.debug_tiles:
                            cv2.imwrite(self.dir_subtiles_results + "{}-{}-{}-{}_empty.jpg".format(big_tile_x, big_tile_y, xi, yi), subtile)

                    subtiles_trees[xi, yi] = subtile_inner_trees

            big_tile_trees = None
            for xi, yi in sorted(subtiles_trees.keys()):
                tox, toy = min(big_tile.shape[1], 2*border+(xi + 1) * tile_inner_size), min(big_tile.shape[0], 2*border+(yi + 1) * tile_inner_size)
                fromx, fromy = tox - self.patch_size, toy - self.patch_size

                a = subtiles_trees[xi, yi]

                a_idx_on_border = []
                for idx, rowA in a.iterrows():
                    axmin, aymin, axmax, aymax, ascore = int(rowA.xmin), int(rowA.ymin), int(rowA.xmax), int(rowA.ymax), rowA.score
                    if axmin > fromx + border and axmax < tox - border and aymin > fromy + border and aymax < toy - border:
                        continue
                    a_idx_on_border.append(idx)

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = xi + dx, yi + dy
                        if (nx, ny) not in subtiles_trees:
                            continue
                        b = subtiles_trees[nx, ny]

                        indices_to_check = a_idx_on_border

                        # because the last two columns/rows have much bigger overlap
                        if    (xi == inner_tiles_nx - 2 and dx == 1) or (xi == inner_tiles_nx - 1 and dx == -1)\
                           or (yi == inner_tiles_ny - 2 and dy == 1) or (yi == inner_tiles_ny - 1 and dy == -1):
                            indices_to_check = a.index

                        for idx in indices_to_check:
                            rowA = a.loc[idx]
                            if rowA.label == "Suppressed":
                                continue
                            axmin, aymin, axmax, aymax, ascore = int(rowA.xmin), int(rowA.ymin), int(rowA.xmax), int(rowA.ymax), rowA.score
                            areaA = (axmax - axmin) * (aymax - aymin)
                            for _, rowB in b.iterrows():
                                bxmin, bymin, bxmax, bymax, bscore = int(rowB.xmin), int(rowB.ymin), int(rowB.xmax), int(rowB.ymax), rowB.score
                                areaB = (bxmax - bxmin) * (bymax - bymin)

                                intersectionx = max(0, min(axmax, bxmax) - max(axmin, bxmin))
                                intersectiony = max(0, min(aymax, bymax) - max(aymin, bymin))
                                intersectionArea = intersectionx * intersectiony
                                if intersectionArea > min(areaA, areaB) * area_overlap_threshold:
                                    if ascore + 0.2 < bscore:
                                        a.loc[idx, 'label'] = "Suppressed"
                                    elif not (bscore + 0.2 < ascore):
                                        if areaA < areaB:
                                            a.loc[idx, 'label'] = "Suppressed"
                                        elif not (areaB < areaA) and (xi, yi) < (nx, ny):
                                            assert not ((nx, ny) < (xi, yi))
                                            a.loc[idx, 'label'] = "Suppressed"

                if big_tile_trees is None:
                    big_tile_trees = pd.DataFrame(columns=a.columns)
                for idx, row in a.iterrows():
                    if row.label == "Suppressed":
                        continue
                    big_tile_trees = pandas_append(big_tile_trees, row, ignore_index=True)

            idx_on_borders = []
            for idx, rowA in big_tile_trees.iterrows():
                axmin, aymin, axmax, aymax, ascore = int(rowA.xmin), int(rowA.ymin), int(rowA.xmax), int(rowA.ymax), rowA.score
                if axmin > 2*border and axmax < big_tiles_k * self.patch_size and aymin > 2*border and aymax < big_tiles_k * self.patch_size:
                    continue
                idx_on_borders.append(idx)

            bigtiles_trees[big_tile_x, big_tile_y] = big_tile_trees
            bigtiles_to_world[big_tile_x, big_tile_y] = big_tile_to_world
            bigtiles_idx_on_borders[big_tile_x, big_tile_y] = idx_on_borders

            if self.debug_tiles:
                cv2.imwrite(self.dir_detection_results + "{}-{}_clean.jpg".format(big_tile_x, big_tile_y), big_tile)
                img_with_trees = self.debug_draw_trees(big_tile, big_tile_trees)
                cv2.imwrite(self.dir_detection_results + "{}-{}_all_trees.jpg".format(big_tile_x, big_tile_y), img_with_trees)
                img_with_border_trees = self.debug_draw_trees(big_tile, big_tile_trees.loc[idx_on_borders])
                cv2.imwrite(self.dir_detection_results + "{}-{}_border_trees.jpg".format(big_tile_x, big_tile_y), img_with_border_trees)

            self.detectionPBar.setValue((big_tile_index + 1) * 100 / len(big_tiles))
            Metashape.app.update()
            app.processEvents()
            self.check_stopped()

        for big_tile_x, big_tile_y in sorted(big_tiles):
            big_tile_trees = bigtiles_trees[big_tile_x, big_tile_y]
            if big_tile_trees is None:
                continue

            big_tile_to_world = bigtiles_to_world[big_tile_x, big_tile_y]

            a_idx_on_borders = bigtiles_idx_on_borders[big_tile_x, big_tile_y]

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = big_tile_x + dx, big_tile_y + dy
                    if (nx, ny) not in bigtiles_trees:
                        continue
                    b = bigtiles_trees[nx, ny]
                    if b is None:
                        continue

                    b_idx_on_borders = bigtiles_idx_on_borders[nx, ny]

                    for idxA in a_idx_on_borders:
                        rowA = big_tile_trees.loc[idxA]
                        if rowA.label == "Suppressed":
                            continue
                        axmin, aymin, axmax, aymax, ascore = int(rowA.xmin), int(rowA.ymin), int(rowA.xmax), int(rowA.ymax), rowA.score
                        areaA = (axmax - axmin) * (aymax - aymin)
                        for idxB in b_idx_on_borders:
                            rowB = b.loc[idxB]
                            bxmin, bymin, bxmax, bymax, bscore = int(rowB.xmin), int(rowB.ymin), int(rowB.xmax), int(rowB.ymax), rowB.score
                            bxmin, bxmax = map(lambda x: x + dx * big_tiles_k * self.patch_size, [bxmin, bxmax])
                            bymin, bymax = map(lambda y: y + dy * big_tiles_k * self.patch_size, [bymin, bymax])
                            areaB = (bxmax - bxmin) * (bymax - bymin)

                            intersectionx = max(0, min(axmax, bxmax) - max(axmin, bxmin))
                            intersectiony = max(0, min(aymax, bymax) - max(aymin, bymin))
                            intersectionArea = intersectionx * intersectiony
                            if intersectionArea > min(areaA, areaB) * area_overlap_threshold:
                                if ascore + 0.2 < bscore:
                                    big_tile_trees.loc[idxA, 'label'] = "Suppressed"
                                elif not (bscore + 0.2 < ascore):
                                    if areaA < areaB:
                                        big_tile_trees.loc[idxA, 'label'] = "Suppressed"
                                    elif not (areaB < areaA) and (big_tile_x, big_tile_y) < (nx, ny):
                                        assert not ((nx, ny) < (big_tile_x, big_tile_y))
                                        big_tile_trees.loc[idxA, 'label'] = "Suppressed"

            big_tile_trees = big_tile_trees[big_tile_trees.label != "Suppressed"]

            if self.debug_tiles:
                big_tile = cv2.imread(self.dir_detection_results + "{}-{}_clean.jpg".format(big_tile_x, big_tile_y))
                img_with_trees = self.debug_draw_trees(big_tile, big_tile_trees)
                cv2.imwrite(self.dir_detection_results + "{}-{}_after_suppression.jpg".format(big_tile_x, big_tile_y), img_with_trees)

            ntrees_detected += len(big_tile_trees)
            self.add_trees(big_tile_to_world, big_tile_trees, detected_shapes_layer)

        self.results_ntrees_detected = ntrees_detected
        self.results_time_detection = time.time() - time_start

    def add_trees(self, to_world, tile_trees, shapes_group):
        import numpy as np

        for row in tile_trees.itertuples():
            xmin, ymin, xmax, ymax, label = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax), row.label
            assert (label == "Tree")

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                x, y = to_world @ np.array([x+0.5, y+0.5, 1]).reshape(3, 1)
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = shapes_group
            shape.geometry = Metashape.Geometry.Polygon(corners)

    def show_results_dialog(self):
        message = "Finished in {:.2f} sec:\n".format(self.results_time_total)\
                   + "{} trees detected.".format(self.results_ntrees_detected)

        print(message)
        Metashape.app.messageBox(message)

    def create_gui(self):
        self.labelTrainZonesLayer = QtWidgets.QLabel("Train zones:")
        self.labelTrainDataLayer = QtWidgets.QLabel("Train data:")

        self.trainZonesLayer = QtWidgets.QComboBox()
        self.trainDataLayer = QtWidgets.QComboBox()
        self.noTrainDataChoice = (None, "No additional training (use pre-trained model as is)", True)

        self.layers = [self.noTrainDataChoice]

        slow_shape_layers_enumerating_but_with_number_of_shapes = False
        if slow_shape_layers_enumerating_but_with_number_of_shapes:
            print("Enumerating all shape layers...")
            shapes_enumerating_start = time.time()
            self.layersDict = {}
            self.layersSize = {}
            shapes = self.chunk.shapes
            for shape in shapes:
                layer = shape.group
                if layer.key not in self.layersDict:
                    self.layersDict[layer.key] = (layer.key, layer.label, layer.enabled)
                    self.layersSize[layer.key] = 1
                else:
                    self.layersSize[layer.key] += 1
            print("Found {} shapes layers in {:.2f} sec:".format(len(self.layersDict), time.time() - shapes_enumerating_start))
            for key in sorted(self.layersDict.keys()):
                key, label, enabled = self.layersDict[key]
                size = self.layersSize[key]
                print("Shape layer: {} shapes, key={}, label={}".format(size, key, label))
                if label == '':
                    label = 'Layer'
                label = label + " ({} shapes)".format(size)
                self.layers.append((key, label, enabled))
            self.layersDict = None
            self.layersSize = None
        else:
            if self.chunk.shapes is None:
                print("No shapes")
            else:
                for layer in self.chunk.shapes.groups:
                    key, label, enabled = layer.key, layer.label, layer.enabled
                    print("Shape layer: key={}, label={}, enabled={}".format(key, label, enabled))
                    if label == '':
                        label = 'Layer'
                    self.layers.append((key, label, layer.enabled))

        for key, label, enabled in self.layers:
            self.trainZonesLayer.addItem(label)
            self.trainDataLayer.addItem(label)
        self.trainZonesLayer.setCurrentIndex(0)
        self.trainDataLayer.setCurrentIndex(0)
        for i, (key, label, enabled) in enumerate(self.layers):
            if not enabled:
                continue
            if label.lower().startswith(self.expected_layer_name_train_zones.lower()):
                self.trainZonesLayer.setCurrentIndex(i)
            if label.lower().startswith(self.expected_layer_name_train_data.lower()):
                self.trainDataLayer.setCurrentIndex(i)

        self.chkUse10cmResolution = QtWidgets.QCheckBox("Process with 10 cm/pix resolution (for speedup)")
        self.chkUse10cmResolution.setToolTip("Process with downsampling to 10 cm/pix instad of original orthomosaic resolution. Leads to faster processing.")
        self.chkUse10cmResolution.setChecked(not self.prefer_original_resolution)

        # self.chkObjectCanBeOfAnyColor = QtWidgets.QCheckBox("An object can be of any color (f.e. in case you want to detect cars)")
        # self.chkObjectCanBeOfAnyColor.setToolTip("If you want to detect cars - tick this checkbox, and even if there are no red cars in train zone,\nneural network will try to detect cars of all colors thanks to this option.")
        # self.chkObjectCanBeOfAnyColor.setChecked(self.augment_colors)

        # self.chkUseBirdsPretrainedModel = QtWidgets.QCheckBox("Use neural network pre-trained on birds")
        # self.chkUseBirdsPretrainedModel.setToolTip("If you want to detect something looking more like a bird (than like a tree) - you can try neural network pre-trained on birds.\n"
        #                                            "You can try both versions to choose the best result, don't forget to annotate small region for addtional training.")

        self.groupBoxGeneral = QtWidgets.QGroupBox("General")
        generalLayout = QtWidgets.QGridLayout()

        self.txtWorkingDir= QtWidgets.QLabel()
        self.txtWorkingDir.setText("Working dir:")
        self.edtWorkingDir= QtWidgets.QLineEdit()
        self.edtWorkingDir.setText(self.working_dir)
        self.edtWorkingDir.setPlaceholderText("Path to dir for intermediate data")
        self.edtWorkingDir.setToolTip("Path to dir for intermediate data")
        self.btnWorkingDir = QtWidgets.QPushButton("...")
        self.btnWorkingDir.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnWorkingDir, QtCore.SIGNAL("clicked()"), lambda: self.choose_working_dir())
        generalLayout.addWidget(self.txtWorkingDir, 0, 0)
        generalLayout.addWidget(self.edtWorkingDir, 0, 1)
        generalLayout.addWidget(self.btnWorkingDir, 0, 2)

        generalLayout.addWidget(self.chkUse10cmResolution, 1, 1)
        self.groupBoxGeneral.setLayout(generalLayout)

        self.groupBoxModelTraining = QtWidgets.QGroupBox("Additional model training (recommended to train at least on a 50x50m zone)")
        trainingLayout = QtWidgets.QGridLayout()

        trainingLayout.addWidget(self.labelTrainZonesLayer, 0, 0)
        trainingLayout.addWidget(self.trainZonesLayer, 0, 1, 1, 2)

        trainingLayout.addWidget(self.labelTrainDataLayer, 1, 0)
        trainingLayout.addWidget(self.trainDataLayer, 1, 1, 1, 2)

        self.groupBoxModelTraining.setLayout(trainingLayout)

        self.groupBoxModelSaveLoad = QtWidgets.QGroupBox("Save/Load trained model (optional, note that orthomosaic resolution should be the same)")
        saveLoadLayout = QtWidgets.QGridLayout()

        self.txtModelSavePath = QtWidgets.QLabel()
        self.txtModelSavePath.setText("Save model to:")
        self.edtModelSavePath = QtWidgets.QLineEdit()
        self.edtModelSavePath.setText(self.save_model_path)
        self.edtModelSavePath.setPlaceholderText("File for neural network model to be saved after additional training")
        self.edtModelSavePath.setToolTip("File for neural network model to be saved after additional training")
        self.btnModelSavePath = QtWidgets.QPushButton("...")
        self.btnModelSavePath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnModelSavePath, QtCore.SIGNAL("clicked()"), lambda: self.choose_model_save_path())
        saveLoadLayout.addWidget(self.txtModelSavePath, 0, 0)
        saveLoadLayout.addWidget(self.edtModelSavePath, 0, 1)
        saveLoadLayout.addWidget(self.btnModelSavePath, 0, 2)
        self.txtModelLoadPath = QtWidgets.QLabel()
        self.txtModelLoadPath.setText("Load model from:")
        self.edtModelLoadPath = QtWidgets.QLineEdit()
        self.edtModelLoadPath.setText(self.load_model_path)
        self.edtModelLoadPath.setPlaceholderText("File with previously saved neural network model (resolution must be the same)")
        self.edtModelLoadPath.setToolTip("File with previously saved neural network model (resolution must be the same)")
        self.btnModelLoadPath = QtWidgets.QPushButton("...")
        self.btnModelLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnModelLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_model_load_path())
        saveLoadLayout.addWidget(self.txtModelLoadPath, 1, 0)
        saveLoadLayout.addWidget(self.edtModelLoadPath, 1, 1)
        saveLoadLayout.addWidget(self.btnModelLoadPath, 1, 2)

        self.groupBoxModelSaveLoad.setLayout(saveLoadLayout)

        self.btnRun = QtWidgets.QPushButton("Run")
        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)

        layout = QtWidgets.QGridLayout()
        row = 0

        layout.addWidget(self.groupBoxGeneral, row, 0, 1, 3)
        row += 1

        layout.addWidget(self.groupBoxModelTraining, row, 0, 1, 3)
        row += 1

        layout.addWidget(self.groupBoxModelSaveLoad, row, 0, 1, 3)
        row += 1

        # layout.addWidget(self.chkObjectCanBeOfAnyColor, row, 1)
        # row += 1

        # layout.addWidget(self.chkUseBirdsPretrainedModel, row, 1)
        # row += 1

        self.txtTrainPBar = QtWidgets.QLabel()
        self.txtTrainPBar.setText("Training progress:")
        self.trainPBar = QtWidgets.QProgressBar()
        self.trainPBar.setTextVisible(True)
        layout.addWidget(self.txtTrainPBar, row, 0)
        layout.addWidget(self.trainPBar, row, 1, 1, 2)
        row += 1

        self.txtDetectionPBar = QtWidgets.QLabel()
        self.txtDetectionPBar.setText("Detection progress:")
        self.detectionPBar = QtWidgets.QProgressBar()
        self.detectionPBar.setTextVisible(True)
        layout.addWidget(self.txtDetectionPBar, row, 0)
        layout.addWidget(self.detectionPBar, row, 1, 1, 2)
        row += 1

        layout.addWidget(self.btnRun, row, 1)
        layout.addWidget(self.btnStop, row, 2)
        row += 1

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnRun, QtCore.SIGNAL("clicked()"), lambda: self.process())
        QtCore.QObject.connect(self.btnStop, QtCore.SIGNAL("clicked()"), lambda: self.stop())

    def choose_working_dir(self):
        working_dir = Metashape.app.getExistingDirectory()
        self.edtWorkingDir.setText(working_dir)

    def choose_model_save_path(self):
        models_dir = ""
        load_path = Metashape.app.settings.value("scripts/detect_objects/model_load_path")
        if load_path is not None:
            models_dir = str(pathlib.Path(load_path).parent)

        save_path = Metashape.app.getSaveFileName("Trained model save path", models_dir, ".model")
        if len(save_path) > 0 and save_path.split(".")[-1] != "model":
            save_path += ".model"

        self.edtModelSavePath.setText(save_path)

    def choose_model_load_path(self):
        load_path = Metashape.app.getOpenFileName("Trained model load path", "", "*.model")
        self.edtModelLoadPath.setText(load_path)

    def readModelLoadPathFromSettings(self):
        load_path = Metashape.app.settings.value("scripts/detect_objects/model_load_path")
        if load_path is None:
            load_path = ""
        return load_path

    def saveToSettingsModelLoadPath(self, load_path):
        Metashape.app.settings.setValue("scripts/detect_objects/model_load_path", load_path)

    def load_params(self):
        import sys

        self.prefer_original_resolution = not self.chkUse10cmResolution.isChecked()

        # self.use_neural_network_pretrained_on_birds = self.chkUseBirdsPretrainedModel.isChecked()

        # self.augment_colors = self.chkObjectCanBeOfAnyColor.isChecked()

        if not self.prefer_original_resolution:
            self.orthomosaic_resolution = self.preferred_resolution
            self.patch_size = self.preferred_patch_size
        else:
            self.orthomosaic_resolution = self.chunk.orthomosaic.resolution
            if self.orthomosaic_resolution > 0.105:
                raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")
            if self.force_small_patch_size:
                patch_size_multiplier = 1
            else:
                patch_size_multiplier = max(1, min(4, self.preferred_resolution / self.orthomosaic_resolution))
            self.patch_size = round(self.preferred_patch_size * patch_size_multiplier)

        self.patch_inner_border = self.patch_size // 8
        print("Using resolution {} m/pix with patch {}x{}".format(self.orthomosaic_resolution, self.patch_size, self.patch_size))
        self.working_dir = self.edtWorkingDir.text()

        self.load_model_path = self.edtModelLoadPath.text()
        self.save_model_path = self.edtModelSavePath.text()

        trainZonesLayer = self.layers[self.trainZonesLayer.currentIndex()]
        trainDataLayer = self.layers[self.trainDataLayer.currentIndex()]
        if trainZonesLayer == self.noTrainDataChoice or trainDataLayer == self.noTrainDataChoice:
            self.train_on_user_data_enabled = False
            print("Additional neural network training disabled")
        else:
            self.train_on_user_data_enabled = True
            print("Additional neural network training expected on key={} layer data w.r.t. key={} layer zones".format(trainDataLayer[0], trainZonesLayer[0]))
            print("Loading train shapes...")
            loading_train_shapes_start = time.time()
            shapes = self.chunk.shapes
            self.train_zones = []
            self.train_data = []
            for shape in shapes:
                layer = shape.group
                if layer.key == trainZonesLayer[0]:
                    self.train_zones.append(shape)
                elif layer.key == trainDataLayer[0]:
                    self.train_data.append(shape)
            print("{} train zones and {} train data loaded in {:.2f} sec".format(len(self.train_zones), len(self.train_data), time.time() - loading_train_shapes_start))

    def debug_draw_trees(self, img, trees):
        import cv2
        import numpy as np
        import pandas as pd

        img = img.copy()

        if isinstance(trees, pd.DataFrame):
            for row in trees.itertuples():
                xmin, ymin, xmax, ymax, label = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax), row.label
                assert label == "Tree"
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        else:
            h, w, cn = img.shape
            for bbox_from, bbox_to in trees:
                assert np.all(bbox_from >= np.int32([0, 0]))
                assert np.all(bbox_to <= np.int32([w, h]))
                (xmin, ymin), (xmax, ymax) = bbox_from, bbox_to
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        return img


def detect_objects():
    chunk = Metashape.app.document.chunk

    if chunk is None or chunk.orthomosaic is None:
        raise Exception("No active orthomosaic.")

    if chunk.orthomosaic.resolution > 0.105:
        raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = DetectObjectsDlg(parent)


label = "Scripts/Detect objects"
Metashape.app.addMenuItem(label, detect_objects)
print("To execute this script press {}".format(label))
