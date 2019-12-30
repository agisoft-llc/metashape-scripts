# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts
#
# Based on https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/style_transfer_3d.ipynb
# Modifications:
# 1. Taking into account cameras positions (when possible) instead of meshutil.sample_view(10.0, 12.0)
# 2. Integration with Metashape Pro to make usage easier
#
# Note that you need to:
# 1. Install CUDA 9.0 and cuDNN for CUDA 9.0
# 2. In Python bundled with Metashape install these packages: tensorflow-gpu==1.9.0 lucid==0.2.3 numpy==1.15.0 Pillow==5.2.0 matplotlib==2.2.2 ipython==6.5.0 PyOpenGL==3.1.0 jupyter==1.0.0
#
# Installation and usage instruction: http://www.agisoft.com/index.php?id=54

import Metashape
import pathlib, shutil, math
from PySide2 import QtGui, QtCore, QtWidgets


# Checking compatibility
compatible_major_version = "1.6"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


class ModelStyleTransferDlg(QtWidgets.QDialog):
    def __init__(self, parent):

        self.texture_size = 2048
        self.rendering_width = 2048
        self.steps_number = 1000
        self.style_path = ""
        self.style_name = "style1"
        self.working_dir = ""
        self.model_name = "model1"
        self.use_cameras_position = len(chunk.cameras) > 0

        self.content_weight = 200.0
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

        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent / "model_style_transfer")
            self.model_name = pathlib.Path(Metashape.app.document.path).stem

        # Paths will be inited in self.exportInput()
        self.input_model_path = None
        self.input_texture_path = None
        self.input_cameras_path = None  # Can be None if no cameras or self.use_cameras_position is False
        self.output_dir = None
        self.output_texture_path = None
        self.result_model_path = None

        # Cameras will be loaded with self.exportCameras() + self.loadCameras() or randomly sampled with meshutil.sample_view(10.0, 12.0)
        self.cameras = None
        self.max_fovy = 10.0
        self.aspect_ratio = 1.0

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Model style transfer")

        self.createGUI()
        self.initDefaultParams()

        self.exec()

    def modelStyleTransfer(self):
        self.loadParams()

        print("Script started...")

        self.exportInput()

        try:
            self.textureStyle3D()
        except:
            Metashape.app.messageBox("Something gone wrong!\n"
                                     "Please check the console.")
            raise
        finally:
            self.reject()

        print("Script finished!")
        return True

    def chooseStylePath(self):
        style_path = Metashape.app.getOpenFileName(filter="*.jpg;;*.jpeg;;*.JPG;;*.JPEG;;*.png;;*.PNG")
        self.edtStylePath.setText(style_path)
        self.edtStyleName.setText(pathlib.Path(style_path).stem)

    def chooseWorkingDir(self):
        working_dir = Metashape.app.getExistingDirectory()
        self.edtWorkingDir.setText(working_dir)

    def createGUI(self):
        layout = QtWidgets.QGridLayout()
        row = 0

        self.txtStylePath= QtWidgets.QLabel()
        self.txtStylePath.setText("Style image:")
        self.txtStylePath.setFixedSize(150, 25)
        self.edtStylePath= QtWidgets.QLineEdit()
        self.edtStylePath.setPlaceholderText("URL or file path")
        self.btnStylePath = QtWidgets.QPushButton("...")
        self.btnStylePath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnStylePath, QtCore.SIGNAL("clicked()"), lambda: self.chooseStylePath())
        layout.addWidget(self.txtStylePath, row, 0)
        layout.addWidget(self.edtStylePath, row, 1)
        layout.addWidget(self.btnStylePath, row, 2)
        row += 1

        self.txtStyleName = QtWidgets.QLabel()
        self.txtStyleName.setText("Style name:")
        self.txtStyleName.setFixedSize(150, 25)
        self.edtStyleName = QtWidgets.QLineEdit()
        layout.addWidget(self.txtStyleName, row, 0)
        layout.addWidget(self.edtStyleName, row, 1, 1, 2)
        row += 1

        self.txtStepsNumber = QtWidgets.QLabel()
        self.txtStepsNumber.setText("Steps number:")
        self.txtStepsNumber.setFixedSize(150, 25)
        self.edtStepsNumber = QtWidgets.QLineEdit()
        self.edtStepsNumber.setPlaceholderText("number of iterations")
        layout.addWidget(self.txtStepsNumber, row, 0)
        layout.addWidget(self.edtStepsNumber, row, 1, 1, 2)
        row += 1

        self.txtTextureSize = QtWidgets.QLabel()
        self.txtTextureSize.setText("Texture size:")
        self.txtTextureSize.setFixedSize(150, 25)
        self.edtTextureSize = QtWidgets.QLineEdit()
        self.edtTextureSize.setPlaceholderText("resulting texture resolution")
        layout.addWidget(self.txtTextureSize, row, 0)
        layout.addWidget(self.edtTextureSize, row, 1, 1, 2)
        row += 1

        self.txtRenderingSize = QtWidgets.QLabel()
        self.txtRenderingSize.setText("Rendering size:")
        self.txtRenderingSize.setFixedSize(150, 25)
        self.edtRenderingSize = QtWidgets.QLineEdit()
        self.edtRenderingSize.setPlaceholderText("width of rendering buffer")
        layout.addWidget(self.txtRenderingSize, row, 0)
        layout.addWidget(self.edtRenderingSize, row, 1, 1, 2)
        row += 1

        self.txtModelName = QtWidgets.QLabel()
        self.txtModelName.setText("Model name:")
        self.txtModelName.setFixedSize(150, 25)
        self.edtModelName = QtWidgets.QLineEdit()
        layout.addWidget(self.txtModelName, row, 0)
        layout.addWidget(self.edtModelName, row, 1, 1, 2)
        row += 1

        self.txtWorkingDir= QtWidgets.QLabel()
        self.txtWorkingDir.setText("Working dir:")
        self.txtWorkingDir.setFixedSize(150, 25)
        self.edtWorkingDir= QtWidgets.QLineEdit()
        self.edtWorkingDir.setPlaceholderText("path to dir")
        self.btnWorkingDir = QtWidgets.QPushButton("...")
        self.btnWorkingDir.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnWorkingDir, QtCore.SIGNAL("clicked()"), lambda: self.chooseWorkingDir())
        layout.addWidget(self.txtWorkingDir, row, 0)
        layout.addWidget(self.edtWorkingDir, row, 1)
        layout.addWidget(self.btnWorkingDir, row, 2)
        row += 1

        self.txtContentWeight= QtWidgets.QLabel()
        self.txtContentWeight.setText("Content weight:")
        self.txtContentWeight.setFixedSize(150, 25)
        self.edtContentWeight= QtWidgets.QLineEdit()
        layout.addWidget(self.txtContentWeight, row, 0)
        layout.addWidget(self.edtContentWeight, row, 1, 1, 2)
        row += 1

        self.txtUseCameraPositions= QtWidgets.QLabel()
        self.txtUseCameraPositions.setText("Use cameras position:")
        self.txtUseCameraPositions.setFixedSize(150, 25)
        self.chbUseCameraPositions= QtWidgets.QCheckBox()
        if len(chunk.cameras) == 0:
            self.chbUseCameraPositions.setEnabled(False)
        layout.addWidget(self.txtUseCameraPositions, row, 0)
        layout.addWidget(self.chbUseCameraPositions, row, 1)
        row += 1

        self.txtPBar = QtWidgets.QLabel()
        self.txtPBar.setText("Progress:")
        self.txtPBar.setFixedSize(150, 25)
        self.pBar = QtWidgets.QProgressBar()
        self.pBar.setTextVisible(False)
        self.pBar.setMinimumSize(239, 25)
        layout.addWidget(self.txtPBar, row, 0)
        layout.addWidget(self.pBar, row, 1, 1, 2)
        row += 1

        self.btnRun = QtWidgets.QPushButton("Run")
        layout.addWidget(self.btnRun, row, 1, 1, 2)
        row += 1

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnRun, QtCore.SIGNAL("clicked()"), lambda: self.modelStyleTransfer())

    def initDefaultParams(self):
        self.edtTextureSize.setText(str(self.texture_size))
        self.edtRenderingSize.setText(str(self.rendering_width))
        self.edtStepsNumber.setText(str(self.steps_number))
        self.edtStylePath.setText(str(self.style_path))
        self.edtStyleName.setText(self.style_name)
        self.edtWorkingDir.setText(self.working_dir)
        self.edtModelName.setText(self.model_name)
        self.edtContentWeight.setText(str(self.content_weight))
        self.chbUseCameraPositions.setChecked(self.use_cameras_position)

    def loadParams(self):
        self.texture_size = int(self.edtTextureSize.text())
        self.rendering_width = int(self.edtRenderingSize.text())
        self.steps_number = int(self.edtStepsNumber.text())
        self.style_path = self.edtStylePath.text()
        self.style_name = self.edtStyleName.text()
        self.working_dir = self.edtWorkingDir.text()
        self.model_name = self.edtModelName.text()
        self.content_weight = float(self.edtContentWeight.text())
        self.use_cameras_position = self.chbUseCameraPositions.isChecked()

        if len(self.style_path) == 0:
            Metashape.app.messageBox("You should specify style image!")
            raise Exception("You should specify style image!")

        if len(self.working_dir) == 0:
            Metashape.app.messageBox("You should specify working dir!")
            raise Exception("You should specify working dir!")

    def exportInput(self):
        working_dir = pathlib.Path(self.working_dir)
        print("Creating working directory '{}'...".format(self.working_dir))
        working_dir.mkdir(parents=True, exist_ok=True)

        self.input_model_path = str(working_dir / "{}.ply".format(self.model_name))
        print("Exporting model to '{}'...".format(self.input_model_path))
        chunk.exportModel(self.input_model_path, binary=True, texture_format=Metashape.ImageFormatJPEG, texture=True,
                          normals=False, colors=False, cameras=False, markers=False, format=Metashape.ModelFormatPLY)
        self.input_model_path = str(working_dir / "{}.obj".format(self.model_name))
        print("Exporting model to '{}'...".format(self.input_model_path))
        chunk.exportModel(self.input_model_path, binary=False, texture_format=Metashape.ImageFormatJPEG, texture=True,
                          normals=False, colors=False, cameras=False, markers=False, format=Metashape.ModelFormatOBJ)

        self.input_texture_path = str(working_dir / "{}.jpg".format(self.model_name))

        self.input_cameras_path = str(working_dir / "{}.cameras".format(self.model_name))
        if not self.use_cameras_position or not self.exportCameras():
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
            if (selection_active and not c.selected) or not c.enabled or c.transform is None or c.type != Metashape.Camera.Type.Regular:
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
        import numpy as np

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

        # if os.name != 'nt':
        #     from lucid.misc.gl.glcontext import create_opengl_context
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

        # if os.name != 'nt':
        #     print("Creating OpenGL context...")
        #     create_opengl_context()
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
        if self.cameras is None:
            mesh = meshutil.normalize_mesh(mesh)

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
            app = QtWidgets.QApplication.instance()

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

                # Show progress
                self.pBar.setValue((i + step_n//10 + 1) / (step_n + step_n//10) * 100)
                app.processEvents()

        reset(style, original_texture)

        print("Running {} iterations...".format(self.steps_number))
        run(mesh, step_n=self.steps_number)

        print("Finished!")
        texture = t_texture.eval()
        print("Exporting result texture to '{}'...".format(self.output_texture_path))
        lucid_io.save(texture, self.output_texture_path, quality=90)

        sess.close()

        print("Importing result model to Metashape '{}'...".format(self.result_model_path))
        chunk.model = None
        chunk.importModel(self.result_model_path)
        chunk.model.label = self.style_name

        Metashape.app.messageBox("Everything worked fine!\n"
                                 "Please save project and RESTART Metashape!\n"
                                 "Because video memory was not released by TensorFlow!")


def model_style_transfer():
    global chunk
    chunk = Metashape.app.document.chunk

    if chunk is None or chunk.model is None:
        raise Exception("No active model!")

    if chunk.model.texture is None or chunk.model.tex_vertices is None or len(chunk.model.tex_vertices) == 0:
        raise Exception("Model is not textured!")

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = ModelStyleTransferDlg(parent)


label = "Custom menu/Model style transfer"
Metashape.app.addMenuItem(label, model_style_transfer)
print("To execute this script press {}".format(label))
