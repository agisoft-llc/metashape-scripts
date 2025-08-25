# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts
#
# ONNX sky segmentation model (skyseg.onnx):
# https://huggingface.co/JianyuanWang/skyseg/blob/main/skyseg.onnx
#
# How to install (Linux):
#
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
#    copy automatic_sky_masking.py script to /home/<username>/.local/share/Agisoft/Metashape Pro/scripts/
# 2. Restart Metashape
#
# How to install (Windows):
#
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
#    copy automatic_sky_masking.py script to C:/Users/<username>/AppData/Local/Agisoft/Metashape Pro/scripts/
# 2. Restart Metashape

import pathlib
import Metashape
import multiprocessing
import concurrent.futures
from modules.pip_auto_install import pip_install

# Checking compatibility
compatible_major_version = "2.2"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

# pip_install('''numpy''')
pip_install("""\
onnxruntime==1.18.1
opencv-python==4.10.0.84
numpy==1.26.4
Pillow==10.3.0
""")
# Model and processing parameters
SKYSEG_ONNX_URL_GLOBAL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
SKYSEG_ONNX_FILE = "skyseg.onnx"  # will be stored next to this script
SKYSEG_PROB_THRESHOLD = 0.5  # probability threshold for sky
SKYSEG_TARGET_CLASS = 1  # if model outputs multiple classes, index of sky class (default 1)

assert 0.0 <= float(SKYSEG_PROB_THRESHOLD) <= 1.0


def generate_automatic_sky_masks_with_onnx(chunk=None):
    try:
        import numpy as np
        import onnxruntime as ort
        import cv2 as cv
        import sys
        from PIL import Image
        import urllib.request
        import ssl
        import shutil
        import os
    except ImportError:
        print(
            "Please ensure that you installed onnxruntime, opencv-python, numpy and Pillow - see instructions in the script")
        raise

    print("Script started...")
    if chunk is None:
        chunk = Metashape.app.document.chunk

        # Ensure model is present (next to script if possible, otherwise user data dir)
        def _get_models_dir():
            try:
                return pathlib.Path(__file__).parent
            except NameError:
                pass
            try:
                from PySide2 import QtCore
                data_root = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DataLocation)
                d = pathlib.Path(data_root) / "models"
                d.mkdir(parents=True, exist_ok=True)
                return d
            except Exception:
                d = pathlib.Path.home() / ".metashape_models"
                d.mkdir(parents=True, exist_ok=True)
                return d

        # URL on HuggingFace must be "resolve", not "blob"
        SKYSEG_ONNX_URL = SKYSEG_ONNX_URL_GLOBAL.replace("/blob/", "/resolve/")

        models_dir = _get_models_dir()
        model_path = models_dir / SKYSEG_ONNX_FILE

        if not model_path.exists():
            print(f"Sky segmentation model not found. Downloading from {SKYSEG_ONNX_URL} ...")
            try:
                ctx = ssl.create_default_context()
                with urllib.request.urlopen(SKYSEG_ONNX_URL, context=ctx) as r, open(model_path, "wb") as f:
                    shutil.copyfileobj(r, f)
            except Exception as e:
                print(f"Failed to download model: {e}", file=sys.stderr)
                raise
            # test that it is not HTML
            if model_path.stat().st_size < 1024:
                raise RuntimeError(f"Downloaded file too small: {model_path} — check URL")
            print(f"Model downloaded to {model_path}")

    cameras = chunk.cameras

    nmasks_exists = 0
    for c in cameras:
        if c.mask is not None:
            nmasks_exists += 1
            print("Camera {} already has mask".format(c.label))
    if nmasks_exists > 0:
        raise Exception("There are already {} masks, please remove them and try again".format(nmasks_exists))

    masks_dirs_created = set()
    cameras_by_masks_dir = {}
    for i, c in enumerate(cameras):
        if not c.type == Metashape.Camera.Type.Regular:  # skip camera track, if any
            continue

        input_image_path = c.photo.path
        image_mask_dir = pathlib.Path(input_image_path).parent / 'masks'
        if image_mask_dir.exists() and str(image_mask_dir) not in masks_dirs_created:
            attempt = 2
            image_mask_dir_attempt = pathlib.Path(str(image_mask_dir) + "_{}".format(attempt))
            while image_mask_dir_attempt.exists() and str(image_mask_dir_attempt) not in masks_dirs_created:
                attempt += 1
                image_mask_dir_attempt = pathlib.Path(str(image_mask_dir) + "_{}".format(attempt))
            image_mask_dir = image_mask_dir_attempt
        if image_mask_dir.exists():
            assert str(image_mask_dir) in masks_dirs_created
        else:
            image_mask_dir.mkdir(parents=False, exist_ok=False)
            masks_dirs_created.add(str(image_mask_dir))
            cameras_by_masks_dir[str(image_mask_dir)] = list()
        cameras_by_masks_dir[str(image_mask_dir)].append(c)

    onnx_lock = multiprocessing.Lock()

    def process_camera(image_mask_dir, c, camera_index):
        if not c.type == Metashape.Camera.Type.Regular:  # skip camera track, if any
            return

        input_image_path = c.photo.path
        print("{}/{} processing: {}".format(camera_index + 1, len(cameras), input_image_path))
        image_mask_name = pathlib.Path(input_image_path).name.split(".")
        if len(image_mask_name) > 1:
            image_mask_name = image_mask_name[:-1]
        image_mask_name = ".".join(image_mask_name)

        image_mask_path = str(image_mask_dir / image_mask_name) + "_mask.png"

        photo_image = c.photo.image()

        image_types_mapping = {'U8': np.uint8, 'U16': np.uint16}
        if photo_image.data_type not in image_types_mapping:
            print("Image type is not supported yet: {}".format(photo_image.data_type))
        if photo_image.cn not in {3, 4}:
            print("Image channels number not supported yet: {}".format(photo_image.cn))
        img = np.frombuffer(photo_image.tostring(), dtype=image_types_mapping[photo_image.data_type]).reshape(
            photo_image.height, photo_image.width, photo_image.cn)[:, :, :3]

        if photo_image.data_type == "U16":
            assert img.dtype == np.uint16
            img = img - np.min(img)
            maxv = np.max(img)
            if maxv > 0:
                img = np.float32(img) * 255.0 / maxv
            else:
                img = np.float32(img)
            img = (img + 0.5).astype(np.uint8)
        assert img.dtype == np.uint8

        # Prepare inference image (resize if necessary)
        h0, w0 = photo_image.height, photo_image.width

        # Initialize ONNX session under lock (GPU providers may require serialized init)
        with onnx_lock:
            providers_available = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers_available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(model_path), providers=providers)

            # Determine input shape
        in_shape = session.get_inputs()[0].shape
        in_h = in_shape[-2] if isinstance(in_shape[-2], int) else None
        in_w = in_shape[-1] if isinstance(in_shape[-1], int) else None

        if in_h is not None and in_w is not None:
            # Fixed-size model
            img_small = Image.fromarray(img).resize((int(in_w), int(in_h)), Image.BILINEAR)
            img_small = np.array(img_small)
        else:
            # Simple downscale for performance (similar spirit to the reference script)
            max_downscale = 4
            min_resolution = 640
            downscale = min(h0 // min_resolution, w0 // min_resolution)
            downscale = min(downscale, max_downscale)
            if downscale > 1:
                img_small = Image.fromarray(img).resize((w0 // downscale, h0 // downscale), Image.BILINEAR)
                img_small = np.array(img_small)
            else:
                img_small = img

                # Preprocess (RGB -> normalize -> NCHW)
        x = img_small.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)

        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: x})[0]

        # Convert output to probability of sky (H, W) in [0,1]
        if out.ndim == 4:
            out = out[0]
        if out.ndim == 3:
            C, H, W = out.shape
            if C == 1:
                logits = out[0]
                if logits.min() < 0 or logits.max() > 1:
                    prob_lr = 1.0 / (1.0 + np.exp(-logits))
                else:
                    prob_lr = logits.astype(np.float32)
            else:
                m = np.max(out, axis=0, keepdims=True)
                e = np.exp(out - m)
                sm = e / (np.sum(e, axis=0, keepdims=True) + 1e-8)
                ch = SKYSEG_TARGET_CLASS
                if not isinstance(ch, int) or ch < 0 or ch >= C:
                    ch = 1 if C >= 2 else 0
                prob_lr = sm[ch, :, :].astype(np.float32)
        elif out.ndim == 2:
            logits = out
            if logits.min() < 0 or logits.max() > 1:
                prob_lr = 1.0 / (1.0 + np.exp(-logits))
            else:
                prob_lr = logits.astype(np.float32)
        else:
            print("Unexpected model output shape: {}".format(out.shape), file=sys.stderr)
            return

            # Resize probability to original image size
        prob_img = Image.fromarray((prob_lr * 255.0).astype(np.uint8)).resize((w0, h0), Image.BILINEAR)
        prob = np.array(prob_img).astype(np.float32) / 255.0

        # Threshold and simple morphology (dilate then erode) to clean edges
        mask = (prob <= float(SKYSEG_PROB_THRESHOLD)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=3)
        mask = cv.erode(mask, kernel, iterations=3)
        mask = np.dstack([mask, mask, mask])

        Image.fromarray(mask).save(image_mask_path)

    with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
        camera_offset = 0
        futures = []
        for masks_dir, dir_cameras in cameras_by_masks_dir.items():
            for camera_index, c in enumerate(dir_cameras):
                futures.append(
                    executor.submit(process_camera, pathlib.Path(masks_dir), c, camera_offset + camera_index))
                Metashape.app.update()
            camera_offset += len(dir_cameras)

        for future in futures:
            concurrent.futures.wait([future])
            future.result()  # to check for exceptions
            Metashape.app.update()

    print("{} masks generated in {} directories:".format(len(cameras), len(masks_dirs_created)))
    for mask_dir in sorted(masks_dirs_created):
        print(mask_dir)

    print("Importing masks into project...")
    for masks_dir, dir_cameras in cameras_by_masks_dir.items():
        chunk.generateMasks(path=masks_dir + "/{filename}_mask.png", masking_mode=Metashape.MaskingMode.MaskingModeFile,
                            cameras=dir_cameras)

    print("Script finished.")


label = "Scripts/Automatic sky masking"

Metashape.app.addMenuItem(label, generate_automatic_sky_masks_with_onnx)
print("To execute this script press {}".format(label))
