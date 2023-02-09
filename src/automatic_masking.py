# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts
#
# Based on https://github.com/danielgatis/rembg (tested on rembg==2.0.24)
#
# See also examples of rembg masking:
# - https://peterfalkingham.com/2021/07/19/rembg-a-phenomenal-ai-based-background-remover/
# - https://www.reddit.com/r/photogrammetry/comments/ogf9ei/metashape_rembg_ml_background_removal_and/
#
# How to install (Linux):
#
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
#    copy automatic_masking.py script to /home/<username>/.local/share/Agisoft/Metashape Pro/scripts/
# 2. Restart Metashape
#
# How to install (Windows):
#
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
#    copy automatic_masking.py script to C:/Users/<username>/AppData/Local/Agisoft/Metashape Pro/scripts/
# 2. Restart Metashape

import pathlib
import Metashape
import multiprocessing
import concurrent.futures
from modules.pip_auto_install import pip_install

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

pip_install('''rembg==2.0.*''')

def generate_automatic_background_masks_with_rembg(chunk=None):
    try:
        import rembg
        import rembg.bg
        import scipy
        import numpy as np
        import io
        from PIL import Image
    except ImportError:
        print("Please ensure that you installed torch and rembg - see instructions in the script")
        raise

    print("Script started...")
    if chunk is None:
        chunk = Metashape.app.document.chunk

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
        if not c.type == Metashape.Camera.Type.Regular: #skip camera track, if any
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

    torch_lock = multiprocessing.Lock()

    def process_camera(image_mask_dir, c, camera_index):
        if not c.type == Metashape.Camera.Type.Regular: #skip camera track, if any
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
        img = np.frombuffer(photo_image.tostring(), dtype=image_types_mapping[photo_image.data_type]).reshape(photo_image.height, photo_image.width, photo_image.cn)[:, :, :3]

        if photo_image.data_type == "U16":
            assert img.dtype == np.uint16
            img = img - np.min(img)
            img = np.float32(img) * 255.0 / np.max(img)
            img = (img + 0.5).astype(np.uint8)
        assert img.dtype == np.uint8

        img = Image.fromarray(img)
        max_downscale = 4
        min_resolution = 640
        downscale = min(photo_image.height // min_resolution, photo_image.width // min_resolution)
        downscale = min(downscale, max_downscale)
        if downscale > 1:
            img = img.resize((photo_image.width // downscale, photo_image.height // downscale))
        img = np.array(img)

        with torch_lock:
            mask = rembg.remove(img, alpha_matting=False, only_mask=True)
        mask = Image.fromarray(mask).resize((photo_image.width, photo_image.height))
        mask = np.array(mask)

        mask = (mask > 10)
        mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=3)
        mask = scipy.ndimage.morphology.binary_erosion(mask, iterations=3)
        mask = mask.astype(np.uint8) * 255
        mask = np.dstack([mask, mask, mask])

        Image.fromarray(mask).save(image_mask_path)

    with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
        camera_offset = 0
        futures = []
        for masks_dir, dir_cameras in cameras_by_masks_dir.items():
            for camera_index, c in enumerate(dir_cameras):
                futures.append(executor.submit(process_camera, pathlib.Path(masks_dir), c, camera_offset + camera_index))
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
        chunk.generateMasks(path=masks_dir + "/{filename}_mask.png", masking_mode=Metashape.MaskingMode.MaskingModeFile, cameras=dir_cameras)

    print("Script finished.")


label = "Scripts/Automatic background masking"

Metashape.app.addMenuItem(label, generate_automatic_background_masks_with_rembg)
print("To execute this script press {}".format(label))
