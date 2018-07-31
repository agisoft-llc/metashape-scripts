# This is python script for PhotoScan Pro. Scripts repository: https://github.com/agisoft-llc/photoscan-scripts
#
# Based on https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/style_transfer_3d.ipynb
# Modifications:
# 1. Taking into account cameras positions (when possible) instead of meshutil.sample_view(10.0, 12.0)
# 2. Integration with PhotoScan Pro to make usage easier
#
# Note that you need to:
# 1. Install CUDA 9.0 and cuDNN for CUDA 9.0
# 2. In Python bundled with PhotoScan install these packages: tensorflow-gpu==1.9.0 lucid==0.2.3 numpy==1.15.0 Pillow==5.2.0 matplotlib==2.2.2 ipython==6.5.0 PyOpenGL==3.1.0 jupyter==1.0.0
# 3. In photoscan.sh add this line:
#      LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH
#    right before this line:
#      LD_LIBRARY_PATH=$dirname:$dirname/python/lib:$LD_LIBRARY_PATH
#
# (Tutorial is coming)

import PhotoScan
import numpy as np
import pathlib, shutil, math
from PySide2 import QtGui, QtCore, QtWidgets


# Checking compatibility
compatible_major_version = "1.4"
found_major_version = ".".join(PhotoScan.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible PhotoScan version: {} != {}".format(found_major_version, compatible_major_version))


class ModelStyleTransferDlg(QtWidgets.QDialog):
    def __init__(self, parent):

        self.inited = False

        self.texture_size = 2048
        self.rendering_width = 2048
        self.steps_number = 10
        self.style_path = "Specify URL or file path to style image"
        self.style_name = "style1"
        self.working_dir = "Specify Directory for intermidiate files"
        self.model_name = "model1"
        self.random_cameras = False

        self.content_weight = 100.0
        self.style_decay = 0.95

        self.googlenet_style_layers = [
            'conv2d2',
            'mixed3a',
            'mixed3b',
            'mixed4a',
            'mixed4b',
            'mixed4c',
        ]
        self.googlenet_content_layer = 'mixed3b'

        if len(PhotoScan.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(PhotoScan.app.document.path).parent / "model_style_transfer")
            self.model_name = pathlib.Path(PhotoScan.app.document.path).stem

        # Paths will be inited in self.exportInput()
        self.input_model_path = None
        self.input_texture_path = None
        self.input_cameras_path = None  # Can be None if no cameras or self.random_cameras is True
        self.output_dir = None
        self.output_texture_path = None
        self.result_model_path = None

        # Cameras will be loaded with self.exportCameras() + self.loadCameras() or randomly sampled with meshutil.sample_view(10.0, 12.0)
        self.cameras = None
        self.max_fovy = 10.0
        self.aspect_ratio = 1.0

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Model style transfer")

        self.btnQuit = QtWidgets.QPushButton("Close")
        self.btnQuit.setFixedSize(90, 50)

        self.btnRun = QtWidgets.QPushButton("Run")
        self.btnRun.setFixedSize(90, 50)

        layout = QtWidgets.QGridLayout()  # creating layout
        layout.addWidget(self.btnRun, 1, 0)
        layout.addWidget(self.btnQuit, 1, 1)

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnRun, QtCore.SIGNAL("clicked()"), lambda: self.modelStyleTransfer())
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()

    def modelStyleTransfer(self):

        print("Script started...")

        self.exportInput()

        try:
            self.textureStyle3D()
        finally:
            self.reject()

        print("Script finished!")
        return True

    def exportInput(self):
        working_dir = pathlib.Path(self.working_dir)
        print("Creating working directory '{}'...".format(self.working_dir))
        working_dir.mkdir(parents=True, exist_ok=True)

        self.input_model_path = str(working_dir / "{}.ply".format(self.model_name))
        print("Exporting model to '{}'...".format(self.input_model_path))
        chunk.exportModel(self.input_model_path, binary=True, texture_format=PhotoScan.ImageFormatJPEG, texture=True,
                          normals=False, colors=False, cameras=False, markers=False, format=PhotoScan.ModelFormatPLY)
        self.input_model_path = str(working_dir / "{}.obj".format(self.model_name))
        print("Exporting model to '{}'...".format(self.input_model_path))
        chunk.exportModel(self.input_model_path, binary=False, texture_format=PhotoScan.ImageFormatJPEG, texture=True,
                          normals=False, colors=False, cameras=False, markers=False, format=PhotoScan.ModelFormatOBJ)

        self.input_texture_path = str(working_dir / "{}.jpg".format(self.model_name))

        self.input_cameras_path = str(working_dir / "{}.cameras".format(self.model_name))
        if self.random_cameras or not self.exportCameras():
            self.input_cameras_path = None

        self.output_dir = working_dir / self.style_name
        print("Creating output directory '{}'...".format(str(self.output_dir)))
        if self.output_dir.exists():
            print("  output directory already exists! Deleting...")
            shutil.rmtree(str(self.output_dir))
        self.output_dir.mkdir(parents=False, exist_ok=False)

        for ext in ["obj", "ply", "mtl"]:
            input_path = working_dir / "{}.{}".format(self.model_name, ext)
            output_path = self.output_dir / "{}.{}".format(self.model_name, ext)
            print("  copying {}.{} to output...".format(self.model_name, ext))
            shutil.copyfile(str(input_path), str(output_path))

        self.output_texture_path = str(self.output_dir / "{}.jpg".format(self.model_name))

        self.result_model_path = str(self.output_dir / "{}.obj".format(self.model_name))

    def exportCameras(self):
        matrices = []
        selection_active = len([c for c in chunk.cameras if c.selected]) > 0
        for c in chunk.cameras:
            if (selection_active and not c.selected) or not c.enabled or c.transform is None:
                continue
            calibration = c.sensor.calibration
            f, w, h = calibration.f, calibration.width, calibration.height
            transformToWorld = chunk.transform.matrix * c.transform
            matrices.append({
                "transformToWorld": eval(str(transformToWorld)[len("Matrix("):-1]),
                "fovH": 2 * math.atan(w / 2 / f) * 180 / math.pi,
                "fovV": 2 * math.atan(h / 2 / f) * 180 / math.pi,
                "w": w,
                "h": h,
            })

        if len(matrices) == 0:
            return False

        with open(self.input_cameras_path, "w") as f:
            f.writelines(str(matrices))

        return True

    def loadCameras(self):
        if self.input_cameras_path is None:
            return None

        with open(self.input_cameras_path) as f:
            self.cameras = f.readline()
        self.cameras = eval(self.cameras)
        if len(self.cameras) == 0:
            print("Cameras will be randomly sampled!")
            self.cameras = None
            self.max_fovy = 10.0
            self.aspect_ratio = 1.0
        else:
            print("Loaded {} cameras!".format(len(self.cameras)))
            self.max_fovy = 0.0
            self.aspect_ratio = 0.0
            for i in range(len(self.cameras)):
                m = np.float32(self.cameras[i]["transformToWorld"])
                m = np.linalg.inv(m)
                m[1, :] = -m[1, :]
                m[2, :] = -m[2, :]
                self.cameras[i]["transformToCamera"] = m
                self.cameras[i]["transformToWorld"] = np.linalg.inv(m)
                self.max_fovy = max(self.cameras[i]["fovV"], self.max_fovy)
                self.aspect_ratio = self.cameras[i]["w"] / self.cameras[i]["h"]
        print("Vertical field of view: {:.2f} degrees. Aspect ratio width/height: {:.2f}.".format(self.max_fovy,
                                                                                                  self.aspect_ratio))

    def textureStyle3D(self):
        print("Importing tensorflow...")
        import tensorflow as tf

        print("Checking that GPU is visible for tensorflow...")
        if not tf.test.is_gpu_available():
            raise Exception("No GPU available for tensorflow!")

        print("Importing other libraries...")
        import os
        import io
        import sys
        from string import Template
        from pathlib import Path

        import numpy as np
        import PIL.Image
        # import matplotlib.pylab as pl

        from IPython.display import clear_output, display, Image, HTML

        from lucid.misc.gl.glcontext import create_opengl_context
        import OpenGL.GL as gl

        from lucid.misc.gl import meshutil
        from lucid.misc.gl import glrenderer
        import lucid.misc.io.showing as show
        import lucid.misc.io as lucid_io
        from lucid.misc.tfutil import create_session

        from lucid.modelzoo import vision_models
        from lucid.optvis import objectives
        from lucid.optvis import param
        from lucid.optvis.style import StyleLoss, mean_l1_loss
        from lucid.optvis.param.spatial import sample_bilinear

        print("Creating OpenGL context...")
        create_opengl_context()
        gl.glGetString(gl.GL_VERSION)

        print("Loading vision model...")
        model = vision_models.InceptionV1()
        model.load_graphdef()

        def prepare_image(fn, size=None):
            data = lucid_io.reading.read(fn)
            im = PIL.Image.open(io.BytesIO(data)).convert('RGB')
            if size:
                im = im.resize(size, PIL.Image.ANTIALIAS)
            return np.float32(im) / 255.0

        self.loadCameras()

        print("Loading input model from '{}'...".format(self.input_model_path))
        mesh = meshutil.load_obj(self.input_model_path)

        print("Loading input texture from '{}'...".format(self.input_texture_path))
        original_texture = prepare_image(self.input_texture_path, (self.texture_size, self.texture_size))

        print("Loading style from '{}'...".format(self.style_path))
        style = prepare_image(self.style_path)

        rendering_width = self.rendering_width
        rendering_height = int(rendering_width // self.aspect_ratio)

        print("Creating renderer with resolution {}x{}...".format(rendering_width, rendering_height))
        renderer = glrenderer.MeshRenderer((rendering_width, rendering_height))
        if self.cameras is not None:
            print("  renderer fovy: {:.2f} degrees".format(self.max_fovy))
            renderer.fovy = self.max_fovy

        sess = create_session(timeout_sec=0)

        # t_fragments is used to feed rasterized UV coordinates for the current view.
        # Channels: [U, V, _, Alpha]. Alpha is 1 for pixels covered by the object, and
        # 0 for background.
        t_fragments = tf.placeholder(tf.float32, [None, None, 4])
        t_uv = t_fragments[..., :2]
        t_alpha = t_fragments[..., 3:]

        # Texture atlas to optimize
        t_texture = param.image(self.texture_size, fft=True, decorrelate=True)[0]

        # Variable to store the original mesh texture used to render content views
        content_var = tf.Variable(tf.zeros([self.texture_size, self.texture_size, 3]), trainable=False)

        # Sample current and original textures with provided pixel data
        t_joined_texture = tf.concat([t_texture, content_var], -1)
        t_joined_frame = sample_bilinear(t_joined_texture, t_uv) * t_alpha
        t_frame_current, t_frame_content = t_joined_frame[..., :3], t_joined_frame[..., 3:]
        t_joined_frame = tf.stack([t_frame_current, t_frame_content], 0)

        # Feeding the rendered frames to the Neural Network
        t_input = tf.placeholder_with_default(t_joined_frame, [None, None, None, 3])
        model.import_graph(t_input)

        # style loss
        style_layers = [sess.graph.get_tensor_by_name('import/%s:0' % s)[0] for s in self.googlenet_style_layers]
        # L1-loss seems to be more stable for GoogleNet
        # Note that we use style_decay>0 to average style-describing Gram matrices
        # over the recent viewports. Please refer to StyleLoss for the details.
        sl = StyleLoss(style_layers, self.style_decay, loss_func=mean_l1_loss)

        # content loss
        content_layer = sess.graph.get_tensor_by_name('import/%s:0' % self.googlenet_content_layer)
        content_loss = mean_l1_loss(content_layer[0], content_layer[1]) * self.content_weight

        # setup optimization
        total_loss = content_loss + sl.style_loss
        t_lr = tf.constant(0.05)
        trainer = tf.train.AdamOptimizer(t_lr)
        train_op = trainer.minimize(total_loss)

        init_op = tf.global_variables_initializer()
        loss_log = []

        def reset(style_img, content_texture):
            del loss_log[:]
            init_op.run()
            sl.set_style({t_input: style_img[None, ...]})
            content_var.load(content_texture)

        def sample_random_view():
            if self.cameras is None:
                return meshutil.sample_view(10.0, 12.0)
            else:
                rand_m = self.cameras[np.random.randint(0, len(self.cameras))]["transformToCamera"].copy()
                return rand_m

        def run(mesh, step_n=400):
            for i in range(step_n):
                fragments = renderer.render_mesh(
                    modelview=sample_random_view(),
                    position=mesh['position'], uv=mesh['uv'],
                    face=mesh['face'])
                _, loss = sess.run([train_op, [content_loss, sl.style_loss]], {t_fragments: fragments})
                loss_log.append(loss)
                if i == 0 or (i + 1) % 50 == 0:
                    # clear_output()
                    last_frame, last_content = sess.run([t_frame_current, t_frame_content], {t_fragments: fragments})
                    # show.images([last_frame, last_content], ['current frame', 'content'])
                if i == 0 or (i + 1) % 10 == 0:
                    print(len(loss_log), loss)
                    pass

                # PhotoScan stuff
                app.processEvents()
                self.pBar.setValue(int(processed / len(mask_list) / len(chunk.frames) * 100))

        reset(style, original_texture)

        print("Running {} iterations...".format(self.steps_number))
        run(mesh, step_n=self.steps_number)

        print("Finished!")
        texture = t_texture.eval()
        print("Exporting result texture to '{}'...".format(self.output_texture_path))
        lucid_io.save(texture, self.output_texture_path, quality=90)

        sess.close()

        print("Importing result model to PhotoScan '{}'...".format(self.result_model_path))
        chunk.model = None
        chunk.importModel(self.result_model_path)

        PhotoScan.app.messageBox("Everything worked fine, but please save project and RESTART PhotoScan,"
                                 " because video memory was not released!")


def model_style_transfer():
    global chunk
    chunk = PhotoScan.app.document.chunk

    if chunk is None or chunk.model is None:
        raise Exception("No active model!")

    if chunk.model.texture is None or chunk.model.tex_vertices is None or len(chunk.model.tex_vertices) == 0:
        raise Exception("Model is not textured!")

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = ModelStyleTransferDlg(parent)


label = "Custom menu/Model style transfer"
PhotoScan.app.addMenuItem(label, model_style_transfer)
print("To execute this script press {}".format(label))
