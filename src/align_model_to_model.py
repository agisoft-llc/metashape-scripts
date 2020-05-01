# Automatically aligns/registers one model/dense point cloud to another one.
#
# This is useful for alignment of two independently photogrammetry-reconstructed models/dense point clouds, or two LIDAR clouds with good overlap.
#
# Also this should work in case of registration between LIDAR points clouds and photogrammetry model,
# but please note that they should cover similar part of surface and you should specify accurate scale ratio.
#
# Alternatively - you can use selection+crop to remove from both point clouds/models everything except object part that presented in both entities.
# And then align with automatic scale+resolution+registration.
# Just don't forget to revert cropping after alignment execution to recover deleted parts.
# After such common-parts-based alignment it should be a good idea to finally optimize alignment for full entities without selection+cropping via option 'Use initial alignment'.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
from PySide2 import QtGui, QtCore, QtWidgets

import os, copy, time, itertools, tempfile
from pathlib import Path

try:
    # Requirements:
    # open3d >= 0.8.0.0  (for points cloud global registration and ICP-based refinement)
    # pyhull >= 2015.2.1 (for automatic scale ratio recognition between two models of the same closed object)
    # On windows see also: https://www.agisoft.com/forum/index.php?topic=11387.msg54281#msg54281
    import open3d as o3d
    from pyhull.convex_hull import ConvexHull
    import numpy as np
except ImportError:
    print("Please ensure that you installed open3d and pyhull via 'pip install open3d pyhull' - see https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-metashape-professional-package")
    print("On windows please see also: https://www.agisoft.com/forum/index.php?topic=11387.msg54281#msg54281")
    raise

# Checking compatibility
compatible_major_version = "1.6"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def align_two_point_clouds(points1_source, points2_target, scale_ratio=None, target_resolution=None, no_global_alignment=False, preview_intermidiate_alignment=True):
    # For example let:
    #  - points2_target - tree with height1=10 and resolution1=0.1 (in its coordinates system)
    #  - points2_target - the same tree but with height2=50 (because of another coordinates system) and resolution2=1.0
    # Then:
    #  - scale_ratio should be height2/height1=50/10=5 or (if scale_ratio=None) it will be guessed based on convex hulls (this works good only for closed objects without noise - like furniture object or house without ground surface around it)
    #  - target_resolution=resolution2=1.0 or (if target_resolution=None) it will be guessed as rough average distance between two points
    #
    # So if you want to align two models/point clouds with not-100% overlap (or for not closed objects) - you should measure and specify scale ratio
    # (note that between LIDAR point clouds scale ratio is mostly 1.0)

    assert(isinstance(points1_source, np.ndarray) and isinstance(points2_target, np.ndarray))
    assert(points1_source.shape[1] == points2_target.shape[1] == 3)
    v1, v2 = points1_source, points2_target

    c1 = np.mean(v1, axis=0)
    c2 = np.mean(v2, axis=0)
    if no_global_alignment:
        c1[:] = 0.0
        c2[:] = 0.0

    v1 = v1 - c1
    v2 = v2 - c2

    if scale_ratio is None:
        if no_global_alignment:
            scale_ratio = 1.0
        else:
            print("Warning! No scale ratio!")
            print("It will be estimated based on convex hulls of point clouds/models and so alignment may fail if object is not closed!")
            print("So if alignment will fail - please manually measure and specify scale ratio!")
            start = time.time()
            v1_subsampled = subsample_points(v1, 100000)
            v2_subsampled = subsample_points(v2, 100000)
            hull_size1 = estimate_convex_hull_size(v1_subsampled)
            hull_size2 = estimate_convex_hull_size(v2_subsampled)
            scale_ratio = hull_size2 / hull_size1
            print("    scale_ratio={} (size1={}, size2={})".format(scale_ratio, hull_size1, hull_size2))
            print("    estimated in {} s".format(time.time() - start))
    if target_resolution is None:
        print("Warning! No target resolution!")
        print("It will be estimated based on rough average distance between points!")
        start = time.time()
        v1_subsampled = subsample_points(v1, 1000)
        source_resolution = 1.5 * estimate_resolution(v1_subsampled) / np.sqrt(len(v1) / len(v1_subsampled)) * scale_ratio
        v2_subsampled = subsample_points(v2, 1000)
        target_resolution = 1.5 * estimate_resolution(v2_subsampled) / np.sqrt(len(v2) / len(v2_subsampled))
        resolution = np.max([source_resolution, target_resolution])
        print("    target_resolution={} (resolution1={}, resolution2={})".format(resolution, source_resolution, target_resolution))
        print("    estimated in {} s".format(time.time() - start))
        target_resolution = resolution

    print("scale_ratio={} target_resolution={}".format(scale_ratio, target_resolution))
    Metashape.app.update()

    v1 = v1 * scale_ratio

    stage = 0
    total_stages = 2 if no_global_alignment else 3

    if no_global_alignment:
        transformation = np.eye(4)
        if preview_intermidiate_alignment:
            print("Initial objects shown!")
            draw_registration_result(v1, v2, title="Initial alignment")
    else:
        stage += 1
        print("{}/{}: Global registration...".format(stage, total_stages))
        start = time.time()
        source_down1, target_down1, global_registration_result = global_registration(v1, v2, global_voxel_size=64.0 * target_resolution)
        print("    estimated in {} s".format(time.time() - start))
        Metashape.app.update()
        if preview_intermidiate_alignment:
            print("{}/{}: Global registration shown!".format(stage, total_stages))
            draw_registration_result(source_down1, target_down1, global_registration_result.transformation, title="Initial global alignment")
        transformation = global_registration_result.transformation

    downscale1 = 8.0
    stage += 1
    print("{}/{}: Coarse ICP registration...".format(stage, total_stages))
    start = time.time()
    icp_voxel_size1 = downscale1 * target_resolution
    source_down1 = downscale_point_cloud(to_point_cloud(v1), icp_voxel_size1)
    target_down1 = downscale_point_cloud(to_point_cloud(v2), icp_voxel_size1)
    icp_result1 = icp_registration(source_down1, target_down1, voxel_size=icp_voxel_size1, transform_init=transformation, max_iterations=100)
    print("    estimated in {} s".format(time.time() - start))
    Metashape.app.update()
    if preview_intermidiate_alignment:
        print("{}/{}: Coarse ICP registration shown!".format(stage, total_stages))
        draw_registration_result(source_down1, target_down1, icp_result1.transformation, title="Intermidiate ICP alignment")
    transformation = icp_result1.transformation

    downscale2 = 1.0
    stage += 1
    print("{}/{}: Fine ICP registration...".format(stage, total_stages))
    start = time.time()
    icp_voxel_size2 = downscale2 * target_resolution
    icp_result2 = icp_registration(to_point_cloud(v1), to_point_cloud(v2), voxel_size=icp_voxel_size2, transform_init=transformation, max_iterations=100)
    print("    estimated in {} s".format(time.time() - start))
    Metashape.app.update()
    if preview_intermidiate_alignment:
        print("{}/{}: Fine ICP registration shown!".format(stage, total_stages))
        draw_registration_result(v1, v2, icp_result2.transformation, title="Resulting alignment")
    transformation = icp_result2.transformation

    T1 = np.diag([1.0, 1.0, 1.0, 1.0])
    T1[:3, 3] = -c1.reshape(3)

    S = np.diag([scale_ratio, scale_ratio, scale_ratio, 1.0])

    T2 = np.diag([1.0, 1.0, 1.0, 1.0])
    T2[:3, 3] = c2.reshape(3)

    M = np.dot(T2, np.dot(transformation, np.dot(S, T1)))
    print("Estimated transformation matrix:")
    print(M)
    Metashape.app.update()
    return M


def subsample_points(vs, n):
    if len(vs) <= n:
        return vs.copy()
    np.random.seed(len(vs))
    vs = vs.copy()
    np.random.shuffle(vs)
    return vs[:n]


def estimate_convex_hull_size(vs):
    hull = ConvexHull(vs)
    indices = np.unique(np.array(list(itertools.chain.from_iterable(hull.vertices)), dtype=np.uint32))
    hull_vs = vs[indices]
    dists = hull_vs[:, None, :] - hull_vs[None, :, :]
    dists = dists.reshape(-1, 3)
    dists = np.sum(dists * dists, axis=-1)
    size = np.sqrt(np.max(dists))
    return size


def estimate_resolution(vs):
    dists = vs[:, None, :] - vs[None, :, :]
    dists = np.sum(dists * dists, axis=-1)
    dists[dists == 0] = np.max(dists)
    min_dists = np.min(dists, axis=-1)
    resolution = np.sqrt(np.median(min_dists))
    return resolution


def to_point_cloud(vs):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(vs.copy())
    return pc


def downscale_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


def estimate_points_features(pcd_down, voxel_size):
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


def global_registration(v1, v2, global_voxel_size):
    # See http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html#global-registration
    source = to_point_cloud(v1)
    target = to_point_cloud(v2)
    source_down = downscale_point_cloud(source, global_voxel_size)
    target_down = downscale_point_cloud(target, global_voxel_size)
    source_fpfh = estimate_points_features(source_down, global_voxel_size)
    target_fpfh = estimate_points_features(target_down, global_voxel_size)

    distance_threshold = global_voxel_size * 2.0
    max_validation = np.min([len(source_down.points), len(target_down.points)]) // 2
    global_registration_result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, max_validation))
    return source_down, target_down, global_registration_result


def icp_registration(source, target, voxel_size, transform_init, max_iterations):
    # See http://www.open3d.org/docs/release/tutorial/Basic/icp_registration.html#icp-registration
    threshold = 8.0 * voxel_size
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, transform_init,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    return reg_p2p


def draw_registration_result(source, target, transformation=None, title="Visualization"):
    Metashape.app.update()
    if isinstance(source, np.ndarray):
        source = to_point_cloud(source)
    if isinstance(target, np.ndarray):
        target = to_point_cloud(target)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.run()
    vis.destroy_window()


def read_ply(filename):
    with Path(filename).open('rb') as file:
        # ply
        # format binary_little_endian 1.0
        # element vertex 236940
        # property float x
        # property float y
        # property float z
        # element face 473918
        # property list uchar int vertex_indices
        # end_header
        line = file.readline()
        assert (line == b'ply\n')

        line = file.readline()
        assert (line == b'format binary_little_endian 1.0\n')

        line = file.readline()
        if (line.startswith(b"comment ")):
            line = file.readline()

        a, b, c = line.strip().split(b' ')
        assert (a == b'element')
        assert (b == b'vertex')
        c = c.replace(b'\n', b'')
        nvertices = int(c)

        line = file.readline()
        assert (line == b'property float x\n')
        line = file.readline()
        assert (line == b'property float y\n')
        line = file.readline()
        assert (line == b'property float z\n')

        line = file.readline()

        if line != b'end_header\n':
            a, b, c = line.split(b' ')
            assert (a == b'element')
            assert (b == b'face')
            c = c.replace(b'\n', b'')
            nfaces = int(c)

            line = file.readline()
            assert (line == b'property list uchar int vertex_indices\n')

            line = file.readline()
            assert (line == b'end_header\n')
        else:
            nfaces = 0
            assert (line == b'end_header\n')

        vertices = np.fromfile(file, dtype=np.float32, count=3 * nvertices)
        vertices = vertices.reshape(nvertices, 3)

        if nfaces > 0:
            face_type = np.dtype([('n', np.uint8), ('vertices', np.uint32, (3,))])
            faces = np.fromfile(file, dtype=face_type, count=nfaces)
            assert (np.all(faces['n'] == 3))
            faces = faces['vertices']
            assert (faces.dtype == np.uint32)
            assert (faces.shape == (nfaces, 3))
            assert (np.max(faces) == nvertices - 1)
        else:
            faces = None

    return vertices


class AlignModelDlg(QtWidgets.QDialog):

    def __init__(self, parent):

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Align model/dense cloud")

        self.labelFrom = QtWidgets.QLabel("From")
        self.labelTo = QtWidgets.QLabel("To")

        self.objects = []
        self.chunk = Metashape.app.document.chunk
        for model in self.chunk.models:
            label = model.label
            if label == '':
                label = "3D Model"
            label += " ({} faces)".format(len(model.faces))
            is_model = True
            self.objects.append((model.key, is_model, label))
        for dense_cloud in self.chunk.dense_clouds:
            label = dense_cloud.label
            if label == '':
                label = "Dense Cloud"
            label += " ({} points)".format(dense_cloud.point_count)
            is_model = False
            self.objects.append((dense_cloud.key, is_model, label))

        self.fromObject = QtWidgets.QComboBox()
        self.toObject = QtWidgets.QComboBox()
        for (key, is_model, label) in self.objects:
            self.fromObject.addItem(label)
            self.toObject.addItem(label)

        self.txtScaleRatio = QtWidgets.QLabel()
        self.txtScaleRatio.setText("Scale ratio:")
        self.edtScaleRatio = QtWidgets.QLineEdit()
        scale_ratio_tooltip = "If empty - will be guessed automatically (works only for closed objects). If scale is the same (for example in case of LIDAR to LIDAR scale ratio) - set scale ratio to 1.0. If target object is twice as big as source object - set scale ratio to 2.0."
        self.txtScaleRatio.setToolTip(scale_ratio_tooltip)
        self.edtScaleRatio.setToolTip(scale_ratio_tooltip)

        self.txtTargetResolution = QtWidgets.QLabel()
        self.txtTargetResolution.setText("Target resolution:")
        self.edtTargetResolution = QtWidgets.QLineEdit()
        target_resolution_tooltip = "If empty - will be guessed automatically (based of average points density). But for LIDAR it is better to specify it manually - for example to 0.1 meters."
        self.txtTargetResolution.setToolTip(target_resolution_tooltip)
        self.edtTargetResolution.setToolTip(target_resolution_tooltip)

        self.chkUseInitialAlignment = QtWidgets.QCheckBox("Use initial alignment")
        self.chkUseInitialAlignment.setToolTip("Start iterative closest points from current alignment (use this if objects are at least coarsly aligned).")

        self.chkPreview = QtWidgets.QCheckBox("Preview intermediate alignment")
        self.chkPreview.setToolTip("Show point clouds intermediate alignment stages, to continue - just close preview window.")

        self.btnOk = QtWidgets.QPushButton("Ok")
        self.btnOk.setFixedSize(90, 50)
        self.btnOk.setToolTip("Align model/dense cloud to another one")

        self.btnQuit = QtWidgets.QPushButton("Close")
        self.btnQuit.setFixedSize(90, 50)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.labelFrom, 0, 0)
        layout.addWidget(self.fromObject, 0, 1)

        layout.addWidget(self.labelTo, 0, 2)
        layout.addWidget(self.toObject, 0, 3)

        layout.addWidget(self.txtScaleRatio, 1, 0)
        layout.addWidget(self.edtScaleRatio, 1, 1)

        layout.addWidget(self.txtTargetResolution, 1, 2)
        layout.addWidget(self.edtTargetResolution, 1, 3)

        layout.addWidget(self.chkUseInitialAlignment, 2, 1)
        layout.addWidget(self.chkPreview, 2, 3)

        layout.addWidget(self.btnOk, 3, 1)
        layout.addWidget(self.btnQuit, 3, 3)

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnOk, QtCore.SIGNAL("clicked()"), self.align)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()

    def align(self):
        print("Script started...")

        (key1, isModel1, label1) = self.objects[self.fromObject.currentIndex()]
        (key2, isModel2, label2) = self.objects[self.toObject.currentIndex()]

        print("Aligning {} to {}...".format(label1, label2))

        region_size = Metashape.app.document.chunk.region.size
        try:
            tmp1 = tempfile.NamedTemporaryFile(delete=False)
            tmp1.close()
            tmp2 = tempfile.NamedTemporaryFile(delete=False)
            tmp2.close()
            Metashape.app.document.chunk.region.size = Metashape.Vector([0.0, 0.0, 0.0])
            for (key, isModel, filename) in [(key2, isModel2, tmp2.name), (key1, isModel1, tmp1.name)]:
                if isModel:
                    self.chunk.model = None
                    for model in self.chunk.models:
                        if model.key == key:
                            self.chunk.model = model
                    assert(self.chunk.model is not None)
                    self.chunk.exportModel(path=filename, binary=True,
                                      save_texture=False, save_uv=False, save_normals=False, save_colors=False,
                                      save_cameras=False, save_markers=False, save_udim=False, save_alpha=False,
                                      save_comment=False,
                                      format=Metashape.ModelFormatPLY)
                else:
                    self.chunk.dense_cloud = None
                    for dense_cloud in self.chunk.dense_clouds:
                        if dense_cloud.key == key:
                            self.chunk.dense_cloud = dense_cloud
                    assert(self.chunk.dense_cloud is not None)
                    self.chunk.exportPoints(path=filename,
                                       source_data=Metashape.DenseCloudData, binary=True,
                                       save_normals=False, save_colors=False, save_classes=False, save_confidence=False,
                                       save_comment=False,
                                       format=Metashape.PointsFormatPLY)

            v1 = read_ply(tmp1.name)
            v2 = read_ply(tmp2.name)
            os.remove(tmp1.name)
            os.remove(tmp2.name)
            Metashape.app.document.chunk.region.size = region_size
        except:
            os.remove(tmp1.name)
            os.remove(tmp2.name)
            Metashape.app.document.chunk.region.size = region_size
            raise

        print("Vertices number: {}, {}".format(len(v1), len(v2)))

        scale_ratio = None if self.edtScaleRatio.text() == '' else float(self.edtScaleRatio.text())
        target_resolution = None if self.edtTargetResolution.text() == '' else float(self.edtTargetResolution.text())
        no_global_alignment = self.chkUseInitialAlignment.isChecked()
        preview_intermidiate_alignment = self.chkPreview.isChecked()

        M12 = align_two_point_clouds(v1, v2, scale_ratio, target_resolution, no_global_alignment, preview_intermidiate_alignment)

        if isModel1:
            assert(self.chunk.model.key == key1)
            try:
                matrix = self.chunk.transform.matrix
                self.chunk.model.transform(Metashape.Matrix(M12) * matrix)
            except AttributeError:
                nvertices = len(self.chunk.model.vertices)
                matrix = Metashape.Matrix(M12)
                vertices = self.chunk.model.vertices
                for i in range(nvertices):
                    vertices[i].coord = matrix.mulp(vertices[i].coord)
                vertices.resize(nvertices)
            self.chunk.model = None
        else:
            assert(self.chunk.dense_cloud.key == key1)
            matrix = self.chunk.transform.matrix
            if self.chunk.dense_cloud.transform is not None:
                matrix = self.chunk.dense_cloud.transform * matrix
            self.chunk.dense_cloud.transform = Metashape.Matrix(M12) * matrix
            self.chunk.dense_cloud = None

        print("Script finished!")
        self.reject()


def show_alignment_dialog():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = AlignModelDlg(parent)


label = "Custom menu/Align model or dense point cloud"
Metashape.app.addMenuItem(label, show_alignment_dialog)
print("To execute this script press {}".format(label))
