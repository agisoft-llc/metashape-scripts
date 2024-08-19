# Exports undistorted photos, internal/external camera orientations, tie points in a simple COLMAP format.
# Exported COLMAP project can be used as input for Gaussian Splatting. See how to use it - https://github.com/PolarNick239/gaussian-splatting-Windows#requirements
#
# Usecase: photo alignment was done in Metashape, export for Gaussian Splatting is required.
#
# Usage:
# 1. Align photos
# 1.1.   Workflow -> Add Photos...
# 1.1.0. (optional, in case of large cropping)
#        Tools -> Camera Calibration... -> select all calibration groups at the left panel -> Fixed parameters: -> Select... -> check "cx, cy"
# 1.2.   Workflow -> Align Photos... (check "Adaptive camera model fitting" if you have several small calibration groups)
# 2. Run script
# 2.1.   Tools -> Run script... then choose this script, without arguments
# 2.2.   Scripts -> Export Colmap project (for Gaussian Splatting)
# 2.3.   Click Export
# 2.4.   Choose destination folder (for several chunks/frames additional subfolders will be added)
#
# Default parameters for GUI can be changed in ExportSceneParams.__init__
#
# Options:
# -- Enforce zero cx, cy -- output camera calibrations will have zero cx and cy.
#        Should be checked until Gaussian Splatting software considers this parameters.
#        May result in information loss during export (large cropping).
#        To mitigate that effect, do step 1.1.0. and check "Adaptive camera model fitting" at 1.2.
# -- Use localframe -- shift coordinates origin to the center of the bounding box, use localframe rotation at this point
# -- Image quality -- quality of the output undistorted images (jpeg only), min 0, max 100
# -- (advanced) Export images -- you can disable export of the undistorted images
#
# Exported files structure (for all chunks and all frames):
# <chosen_folder>
# |---<chunk_0_folder>
# |   |---<frame_0_folder>
# |   |   |---images
# |   |   |   |---<image_0>
# |   |   |   |---<image_1>
# |   |   |   |---...
# |   |   |---sparse
# |   |       |---0
# |   |           |---cameras.bin
# |   |           |---images.bin
# |   |           |---points3D.bin
# |   |---<frame_1_folder>
# |   |---...
# |---<chunk_1_folder>
# |---...
#

import os
import shutil
import struct
import math
from pathlib import Path
from PySide2 import QtCore, QtGui, QtWidgets


# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


f32 = lambda x: bytes(struct.pack("f", x))
d64 = lambda x: bytes(struct.pack("d", x))
u8  = lambda x: x.to_bytes(1, "little", signed=(x < 0))
u32 = lambda x: x.to_bytes(4, "little", signed=(x < 0))
u64 = lambda x: x.to_bytes(8, "little", signed=(x < 0))
bstr = lambda x: bytes((x + "\0"), "utf-8")

def matrix_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if (tr > 0):
        s = 2 * math.sqrt(tr + 1)
        return Metashape.Vector([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        return Metashape.Vector([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if (m[1, 1] > m[2, 2]):
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        return Metashape.Vector([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        return Metashape.Vector([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])

def get_camera_name(cam):
    return ("{:05d}".format(cam.key)) + "_" + os.path.basename(cam.photo.path)

def clean_dir(folder, confirm_deletion):
    if os.path.exists(folder):
        if confirm_deletion:
            ok = Metashape.app.getBool('Folder "' + folder + '" will be deleted.\nAre you sure you want to continue?')
            if not ok:
                return False
        shutil.rmtree(folder)
    os.mkdir(folder)
    return True

def build_dir_structure(folder, confirm_deletion):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not clean_dir(folder + "images/", confirm_deletion):
        return False
    if not clean_dir(folder + "sparse/", confirm_deletion):
        return False
    os.makedirs(folder + "sparse/0/")
    return True

def get_chunk_dirs(folder, params):
    doc = Metashape.app.document
    chunk_name_stats = {}
    chunk_names = {}

    initial_chunk_selected = doc.chunk.selected
    doc.chunk.selected = True

    for chunk in doc.chunks:
        if not params.all_chunks and not chunk.selected:
            continue
        label = chunk.label
        i = chunk_name_stats[label] = chunk_name_stats.get(label, 0)
        while True:
            name = folder + label + ("" if i == 0 else "_" + str(i)) + "/"
            i += 1
            if name not in chunk_names.values():
                chunk_names[chunk.key] = name
                chunk_name_stats[label] = i
                break

    doc.chunk.selected = initial_chunk_selected

    if not params.all_frames and len(chunk_names) == 1:
        return {chunk_key:folder for chunk_key in chunk_names}

    existed = [name for name in chunk_names.values() if os.path.exists(name)]
    if len(existed) > 0:
        ok = Metashape.app.getBool('These folders will be deleted:\n"' + '"\n"'.join(existed) + '"\nAre you sure you want to continue?')
        if not ok:
            return {}
    for name in existed:
        shutil.rmtree(name)
    return chunk_names

def calib_valid(calib, point):
    reproj = calib.project(calib.unproject(point))
    if not reproj:
        return False
    return (reproj - point).norm() < 1.0

def get_valid_calib_region(calib):
    w = calib.width
    h = calib.height

    left = math.floor(calib.cx + w / 2)
    right = math.floor(calib.cx + w / 2)
    top = math.floor(calib.cy + h / 2)
    bottom = math.floor(calib.cy + h / 2)

    left_set = False
    right_set = False
    top_set = False
    bottom_set = False

    max_dim = max(w, h)
    max_tan = math.hypot(w, h) / calib.f

    step_x = 1
    step_y = 1

    if w > h:
        step_y /= min(1.2, (w / h))
    else:
        step_x /= min(1.2, (h / w))

    for r in range(max_dim):
        if left_set and top_set and right_set and bottom_set:
            break

        next_top = top if top_set else math.floor(calib.cy + h / 2 - r * step_y)
        next_bottom = bottom if bottom_set else math.floor(calib.cy + h / 2 + r * step_y)
        next_left = left if left_set else math.floor(calib.cx + w / 2 - r * step_x)
        next_right = right if right_set else math.floor(calib.cx + w / 2 + r * step_x)

        next_top = max(next_top, 0)
        next_left = max(next_left, 0)
        next_right = min(next_right, w - 1)
        next_bottom = min(next_bottom, h - 1)

        for v in range(2):
            for u in range(2):

                if u == 0 and left_set:
                    continue
                if v == 0 and top_set:
                    continue
                if u == 1 and right_set:
                    continue
                if v == 1 and bottom_set:
                    continue

                corner = Metashape.Vector([next_right if u else next_left, next_bottom if v else next_top])
                corner.x += 0.5
                corner.y += 0.5

                step = Metashape.Vector([step_x if u else -step_x, step_y if v else -step_y])

                prev_corner = Metashape.Vector(corner)
                prev_corner -= step

                pt = calib.unproject(corner)

                pt = Metashape.Vector([pt.x / pt.z, pt.y / pt.z])

                prev_pt = calib.unproject(prev_corner)
                prev_pt = Metashape.Vector([prev_pt.x / prev_pt.z, prev_pt.y / prev_pt.z])

                dif = pt - prev_pt

                if (pt.norm() > max_tan or dif * step <= 0 or not calib_valid(calib, corner)):
                    if u:
                        right_set = True
                    else:
                        left_set = True

                    if v:
                        bottom_set = True
                    else:
                        top_set = True

        if not left_set:
            left = next_left
        if not top_set:
            top = next_top
        if not right_set:
            right = next_right
        if not bottom_set:
            bottom = next_bottom

    right += 1
    bottom += 1

    new_w = right - left
    new_h = bottom - top

    border = math.ceil(0.01 * min(new_w, new_h))

    if left_set:
        left += border
    if right_set:
        right -= border
    if top_set:
        top += border
    if bottom_set:
        bottom -= border

    new_w = right - left
    new_h = bottom - top

    if (new_w != w or new_h != h):
        print("Cropped initial calibration due to high distortion", str(w) + "x" + str(h), "->", str(new_w) + "x" + str(new_h))

    return (top, right, bottom, left)
    
def rotate_vector(vec, axis, angle):
    axis = axis.normalized()
    collinear = axis * (vec * axis)
    orthogonal0 = vec - collinear
    orthogonal1 = Metashape.Vector.cross(axis, orthogonal0)
    return collinear + orthogonal0 * math.cos(angle) + orthogonal1 * math.sin(angle)

def axis_magnitude_rotation(axis):
    angle = axis.norm()
    axis = axis.normalized()
    x = Metashape.Vector((1, 0, 0))
    y = Metashape.Vector((0, 1, 0))
    z = Metashape.Vector((0, 0, 1))
    return Metashape.Matrix((rotate_vector(x, axis, -angle), rotate_vector(y, axis, -angle), rotate_vector(z, axis, -angle)))

def compute_size(top, right, bottom, left, T1):
    T1_inv = T1.inv()

    tl = T1_inv.mulp(Metashape.Vector([left, top, 1]))
    tr = T1_inv.mulp(Metashape.Vector([right, top, 1]))
    bl = T1_inv.mulp(Metashape.Vector([left, bottom, 1]))
    br = T1_inv.mulp(Metashape.Vector([right, bottom, 1]))

    halfwl = min(-tl.x / tl.z, -bl.x / bl.z)
    halfwr = min(tr.x / tr.z, br.x / br.z)
    halfht = min(-tr.y / tr.z, -tl.y / tl.z)
    halfhb = min(br.y / br.z, bl.y / bl.z)

    return (halfht, halfwr, halfhb, halfwl)

def compute_undistorted_calib(sensor, zero_cxy):
    border = 0 # in pixels, can be increased if black margins are on the undistorted images

    if sensor.type != Metashape.Sensor.Type.Frame and sensor.type != Metashape.Sensor.Type.Fisheye:
        return (Metashape.Calibration(), Metashape.Matrix.Diag([1, 1, 1, 1]))

    calib_initial = sensor.calibration
    w = calib_initial.width
    h = calib_initial.height
    f = calib_initial.f

    (reg_top, reg_right, reg_bottom, reg_left) = get_valid_calib_region(calib_initial)

    left = -float("inf")
    right = float("inf")
    top = -float("inf")
    bottom = float("inf")

    for i in range(reg_top, reg_bottom):
        im_pt = Metashape.Vector([reg_left + 0.5, i + 0.5])
        if (calib_valid(calib_initial, im_pt)):
            pt = calib_initial.unproject(im_pt)
            left = max(left, pt.x / pt.z)

        im_pt = Metashape.Vector([reg_right - 0.5, i + 0.5])
        if (calib_valid(calib_initial, im_pt)):
            pt = calib_initial.unproject(im_pt)
            right = min(right, pt.x / pt.z)

    for i in range(reg_left, reg_right):
        im_pt = Metashape.Vector([i + 0.5, reg_top + 0.5])
        if (calib_valid(calib_initial, im_pt)):
            pt = calib_initial.unproject(im_pt)
            top = max(top, pt.y / pt.z)

        im_pt = Metashape.Vector([i + 0.5, reg_bottom - 0.5])
        if (calib_valid(calib_initial, im_pt)):
            pt = calib_initial.unproject(im_pt)
            bottom = min(bottom, pt.y / pt.z)

    if right <= left or bottom <= top:
        return (Metashape.Calibration(), Metashape.Matrix.Diag([1, 1, 1, 1]))

    T1 = Metashape.Matrix.Diag([1, 1, 1, 1])
    if zero_cxy:
        left_ang = math.atan(left)
        right_ang = math.atan(right)
        top_ang = math.atan(top)
        bottom_ang = math.atan(bottom)

        rotation_vec = Metashape.Vector([math.tan((left_ang + right_ang) / 2), math.tan((top_ang + bottom_ang) / 2), 1]).normalized()
        rotation_vec = Metashape.Vector.cross(Metashape.Vector((0, 0, 1)), rotation_vec)
        T1 = Metashape.Matrix.Rotation(axis_magnitude_rotation(rotation_vec))

    (halfht, halfwr, halfhb, halfwl) = compute_size(top, right, bottom, left, T1)

    halfht = math.floor(f * halfht)
    halfwr = math.floor(f * halfwr)
    halfhb = math.floor(f * halfhb)
    halfwl = math.floor(f * halfwl)

    halfw = min(halfwl, halfwr)
    halfh = min(halfht, halfhb)

    if zero_cxy:
        halfwl = halfw
        halfwr = halfw
        halfht = halfh
        halfhb = halfh

    max_dim = max(w, h)

    calib = Metashape.Calibration()
    calib.f = f
    calib.width = min(math.floor(max_dim * 1.2), math.floor(halfwl + halfwr) - 2 * border)
    calib.height = min(math.floor(max_dim * 1.2), math.floor(halfht + halfhb) - 2 * border)
    calib.cx = halfwl - (halfwl + halfwr) / 2
    calib.cy = halfht - (halfht + halfhb) / 2

    return (calib, T1)

def check_undistorted_calib(sensor, calib, T1):
    border = 0 # in pixels, can be increased if black margins are on the undistorted images

    calib_initial = sensor.calibration
    w = calib.width
    h = calib.height

    left = float("inf")
    right = -float("inf")
    top = float("inf")
    bottom = -float("inf")

    for i in range(h):
        pt = calib_initial.project(T1.mulp(calib.unproject(Metashape.Vector([0.5, i + 0.5]))))
        left = min(left, pt.x)
        pt = calib_initial.project(T1.mulp(calib.unproject(Metashape.Vector([w - 0.5, i + 0.5]))))
        right = max(right, pt.x)
    for i in range(w):
        pt = calib_initial.project(T1.mulp(calib.unproject(Metashape.Vector([i + 0.5, 0.5]))))
        top = min(top, pt.y)
        pt = calib_initial.project(T1.mulp(calib.unproject(Metashape.Vector([i + 0.5, h - 0.5]))))
        bottom = max(bottom, pt.y)

    print(left, right, top, bottom)
    if (left < 0.5 or calib_initial.width - 0.5 < right or top < 0.5 or calib_initial.height - 0.5 < bottom):
        print("!!! Wrong undistorted calib")
    else:
        print("Ok:")

def get_coord_transform(frame, use_localframe):
    if not use_localframe:
        return frame.transform.matrix
    if not frame.region:
        print("Null region, using world crs instead of local")
        return frame.transform.matrix
    fr_to_gc  = frame.transform.matrix
    gc_to_loc = frame.crs.localframe(fr_to_gc.mulp(frame.region.center))
    fr_to_loc = gc_to_loc * fr_to_gc
    return (Metashape.Matrix.Translation(-fr_to_loc.mulp(frame.region.center)) * fr_to_loc)

def compute_undistorted_calibs(frame, zero_cxy):
    calibs = {} # { sensor_key: ( sensor, undistorted calibration, undistorted camera transform ) }
    for sensor in frame.sensors:
        (calib, T1) = compute_undistorted_calib(sensor, zero_cxy)

        if (calib.width == 0 or calib.height == 0):
            continue
        calibs[sensor.key] = (sensor, calib, T1)
        #check_undistorted_calib(sensor, calib, T1)

    print("Calibrations:")
    for (s_key, (sensor, calib, T1)) in calibs.items():
        print(sensor.key, calib.f, calib.width, calib.height, calib.cx, calib.cy)

    return calibs

def get_calibs(camera, calibs):
    s_key = camera.sensor.key
    if s_key not in calibs:
        cause = "unsupported" if (camera.sensor.type != Metashape.Sensor.Type.Frame and camera.sensor.type != Metashape.Sensor.Type.Fisheye) else "cropped"
        print("Camera " + camera.label + " (key = " + str(camera.key) + ") has " + cause + " sensor (key = " + str(s_key) + ")")
        return (None, None, None)
    return (calibs[s_key][0].calibration, calibs[s_key][1], calibs[s_key][2])


def get_filtered_track_structure(frame, folder, calibs):
    tie_points = frame.tie_points

    cnt_cropped = 0

    tracks = {} # { track_id: [ point indices, good projections, bad projections ] }; projection = ( camera_key, projection_idx )
    images = {} # { camera_key: [ camera, good projections, bad projections ] }; projection = ( undistored pt in pixels, size, track_id )

    for (i, pt) in enumerate(tie_points.points):
        track_id = pt.track_id
        if track_id not in tracks:
            tracks[track_id] = [[], [], []]

        tracks[track_id][0].append(i)

    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue

        T1_inv = T1.inv()
        camera_entry = [cam, [], []]

        projections = tie_points.projections[cam]
        for (i, proj) in enumerate(projections):
            track_id = proj.track_id
            if track_id not in tracks:
                tracks[track_id] = [[], [], []]

            pt = calib1.project(T1_inv.mulp(calib0.unproject(proj.coord)))

            good = False
            if (pt is not None):
                good = (0 <= pt.x and pt.x < calib1.width and 0 <= pt.y and pt.y < calib1.height)

            place = (1 if good else 2)

            if not good:
                cnt_cropped += 1

            pos = len(camera_entry[place])
            camera_entry[place].append((pt, proj.size, track_id))
            tracks[track_id][place].append((cam.key, pos))

        images[cam.key] = camera_entry

    print("Found", cnt_cropped, "cropped projections")
    return (tracks, images)

def save_undistorted_images(params, frame, folder, calibs):
    print("Exporting images.")
    folder = folder + "images/"
    T = Metashape.Matrix.Diag([1, 1, 1, 1])

    cnt = 0
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        if cam.sensor.key not in calibs:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue

        img = cam.image().warp(calib0, T, calib1, T1)
        name = get_camera_name(cam)
        ext = os.path.splitext(name)[1]
        if ext.lower() in [".jpg", ".jpeg"]:
            c = Metashape.ImageCompression()
            c.jpeg_quality = params.image_quality
            img.save(folder + name, c)
        else:
            img.save(folder + name)
        cnt += 1
    print("Undistorted", cnt, "cameras")

def save_undistorted_masks(params, frame, folder, calibs):
    print("Exporting masks.")
    folder = folder + "masks/"
    if not clean_dir(folder, params.confirm_deletion):
        print("Masks folder already exists, aborting.")
        return False

    T = Metashape.Matrix.Diag([1, 1, 1, 1])

    cnt = 0
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        if cam.sensor.key not in calibs:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue
        if not cam.mask:
            # Skip if image has no mask assigned.
            continue

        mask = cam.mask.image().warp(calib0, T, calib1, T1)
        # , 'U8') # Convert image to single channel grayscale.
        mask = mask.convert("L")
        name = get_camera_name(cam)
        mask.save(str(Path(folder + name).with_suffix('.png')))
        cnt += 1
    print("Undistorted", cnt, "masks")

def save_cameras(params, folder, calibs):
    use_pinhole_model = params.use_pinhole_model
    with open(folder + "sparse/0/cameras.bin", "wb") as fout:
        fout.write(u64(len(calibs)))
        for (s_key, (sensor, calib, T1)) in calibs.items():
            fout.write(u32(s_key))
            fout.write(u32(1 if use_pinhole_model else 0))
            fout.write(u64(calib.width))
            fout.write(u64(calib.height))
            fout.write(d64(calib.f))
            if use_pinhole_model:
                fout.write(d64(calib.f))
            fout.write(d64(calib.cx + calib.width * 0.5))
            fout.write(d64(calib.cy + calib.height * 0.5))
    print("Saved", len(calibs), "calibrations")


# { camera_key: [ camera, good projections, bad projections ] }; projection = ( undistored pt in pixels, size, track_id )

def save_images(params, frame, folder, calibs, tracks, images):
    only_good = params.only_good
    T_shift = get_coord_transform(frame, params.use_localframe)

    with open(folder + "sparse/0/images.bin", "wb") as fout:
        fout.write(u64(len(images)))
        for (cam_key, [camera, good_prjs, bad_prjs]) in images.items():
            (calib0, calib1, T1) = get_calibs(camera, calibs)
            transform = T_shift * camera.transform * T1
            R = transform.rotation().inv()
            T = -1 * (R * transform.translation())
            Q = matrix_to_quat(R)
            fout.write(u32(cam_key))
            fout.write(d64(Q.w))
            fout.write(d64(Q.x))
            fout.write(d64(Q.y))
            fout.write(d64(Q.z))
            fout.write(d64(T.x))
            fout.write(d64(T.y))
            fout.write(d64(T.z))
            fout.write(u32(camera.sensor.key))
            fout.write(bstr(get_camera_name(camera)))

            prjs = (good_prjs if only_good else good_prjs + bad_prjs)
            fout.write(u64(len(prjs)))
            for (pt, size, track_id) in prjs:
                track_id = (track_id if len(tracks[track_id][0]) == 1 else -1)
                fout.write(d64(pt.x))
                fout.write(d64(pt.y))
                fout.write(u64(track_id))
    print("Saved", len(images), "cameras")


# { track_id: [ point indices, good projections, bad projections ] }; projection = ( camera_key, projection_idx )

def save_points(params, frame, folder, calibs, tracks, images):
    only_good = params.only_good
    T = get_coord_transform(frame, params.use_localframe)
    num_pts = len(list(filter(lambda x: len(x[0]) == 1, tracks.values())))

    with open(folder + "sparse/0/points3D.bin", "wb") as fout:
        fout.write(u64(num_pts))
        for (track_id, [points, good_prjs, bad_prjs]) in tracks.items():
            if (len(points) != 1):
                continue
            point = frame.tie_points.points[points[0]]
            pt = T * point.coord
            track = frame.tie_points.tracks[track_id]
            fout.write(u64(track_id))
            fout.write(d64(pt.x))
            fout.write(d64(pt.y))
            fout.write(d64(pt.z))
            fout.write(u8(track.color[0]))
            fout.write(u8(track.color[1]))
            fout.write(u8(track.color[2]))
            fout.write(d64(0))

            num = (len(good_prjs) if only_good else len(good_prjs) + len(bad_prjs))
            fout.write(u64(num))
            for (camera_key, proj_idx) in good_prjs:
                fout.write(u32(camera_key))
                fout.write(u32(proj_idx))

            if not only_good:
                for (camera_key, proj_idx) in bad_prjs:
                    fout.write(u32(camera_key))
                    fout.write(u32(proj_idx + len(images[camera_key][1])))
    print("Saved", num_pts, "points from", len(tracks), "tracks")


class ExportSceneParams():
    def __init__(self):
        # default values for parameters
        self.all_chunks = False
        self.all_frames = False

        self.zero_cxy = True
        self.use_localframe = True
        self.image_quality = 90
        self.export_images = True
        self.export_masks = False
        self.confirm_deletion = True
        self.use_pinhole_model = True
        self.only_good = True

    def log(self):
        print("All chunks:", self.all_chunks)
        print("All frames:", self.all_frames)
        print("Zero cx and cy:", self.zero_cxy)
        print("Use local coordinate frame:", self.use_localframe)
        print("Image quality:", self.image_quality)
        print("Export images:", self.export_images)
        print("Export masks:", self.export_masks)
        print("Confirm deletion:", self.confirm_deletion)
        print("Using pinhole model instead of simple_pinhole:", self.use_pinhole_model)
        print("Using only uncropped projections:", self.only_good)


def export_for_gaussian_splatting(params = ExportSceneParams(), progress = QtWidgets.QProgressBar()):
    log_result = lambda x: print("", x, "-----------------------------------", sep="\n")
    progress.setMinimum(0)
    progress.setMaximum(1000)
    set_progress = lambda x: progress.setValue(int(x * 1000))
    params.log()

    folder = Metashape.app.getExistingDirectory("Output folder")
    if len(folder) == 0:
        log_result("No chosen folder")
        return
    folder = folder + "/"
    print(folder)

    chunk_dirs = get_chunk_dirs(folder, params)
    if len(chunk_dirs) == 0:
        log_result("Aborted")
        return

    chunk_num = len(chunk_dirs)
    for chunk_id, (chunk_key, chunk_folder) in enumerate(chunk_dirs.items()):
        chunk = [ck for ck in Metashape.app.document.chunks if ck.key == chunk_key]
        if (len(chunk) != 1):
            print("Chunk not found, key =", chunk_key)
            continue
        chunk = chunk[0]

        frame_num = len(chunk.frames) if params.all_frames else 1
        prog_step = 1 / chunk_num
        set_progress(prog_step * chunk_id)
        set_progress_frame = lambda n: set_progress(prog_step * (chunk_id + n / frame_num))
        frame_cnt = 0

        for frame_id, frame in enumerate(chunk.frames):
            if not frame.tie_points:
                continue
            if not params.all_frames and not (frame == chunk.frame):
                continue
            set_progress_frame(frame_cnt)
            frame_cnt += 1

            folder = chunk_folder + ("" if frame_num == 1 else "frame_" + str(frame_id).zfill(6) + "/")
            print("\n" + folder)

            if not build_dir_structure(folder, params.confirm_deletion):
                log_result("Aborted")
                return

            calibs = compute_undistorted_calibs(frame, params.zero_cxy)

            if params.export_images:
                save_undistorted_images(params, frame, folder, calibs)
            if params.export_masks:
                save_undistorted_masks(params, frame, folder, calibs)
            save_cameras(params, folder, calibs)


            (tracks, images) = get_filtered_track_structure(frame, folder, calibs)
            save_images(params, frame, folder, calibs, tracks, images)
            save_points(params, frame, folder, calibs, tracks, images)

    set_progress(1)
    log_result("Done")


class CollapsibleGroupBox(QtWidgets.QGroupBox):
    def __init__(self, parent = None):
        QtWidgets.QGroupBox.__init__(self, parent)
        self.toggled.connect(self.onCheckedChanged)
        self.maxHeight = self.maximumHeight()

    def onCheckedChanged(self, is_on):
        if not is_on:
            self.oldSize = self.size()

        for child in self.children():
            if isinstance(child, QtWidgets.QWidget):
                child.setVisible(is_on)

        if is_on:
            self.setMaximumHeight(self.maxHeight)
            self.resize(self.oldSize)
        else:
            self.maxHeight = self.maximumHeight()
            self.setMaximumHeight(QtGui.QFontMetrics(self.font()).height() + 4)


class ExportSceneGUI(QtWidgets.QDialog):

    def run_export(self):
        for button in self.buttons:
            button.setEnabled(False)

        params = ExportSceneParams()
        params.all_chunks = self.radioBtn_allC.isChecked()
        params.all_frames = self.radioBtn_allF.isChecked()
        params.zero_cxy = self.zcxyBox.isChecked()
        params.use_localframe = self.locFrameBox.isChecked()
        params.image_quality = self.imgQualSpBox.value()
        params.export_images = self.expImagesBox.isChecked()
        params.export_masks = self.expMasksBox.isChecked()
        try:
            export_for_gaussian_splatting(params, self.pBar)
        finally:
            self.done(0)

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Export scene in Colmap format:")

        defaults = ExportSceneParams()

        self.btnQuit = QtWidgets.QPushButton("Quit")
        self.btnQuit.setFixedSize(100,25)

        self.btnP1 = QtWidgets.QPushButton("Export")
        self.btnP1.setFixedSize(100,25)

        self.pBar = QtWidgets.QProgressBar()
        self.pBar.setTextVisible(False)
        self.pBar.setFixedSize(100, 25)

        self.chnkTxt = QtWidgets.QLabel()
        self.chnkTxt.setText("Chunks:")
        self.chnkTxt.setFixedSize(100, 25)

        self.frmsTxt = QtWidgets.QLabel()
        self.frmsTxt.setText("Frames:")
        self.frmsTxt.setFixedSize(100, 25)

        self.chunk_group = QtWidgets.QButtonGroup()
        self.radioBtn_allC = QtWidgets.QRadioButton("all chunks")
        self.radioBtn_selC = QtWidgets.QRadioButton("selected")
        self.chunk_group.addButton(self.radioBtn_selC)
        self.chunk_group.addButton(self.radioBtn_allC)
        self.radioBtn_allC.setChecked(defaults.all_chunks)
        self.radioBtn_selC.setChecked(not defaults.all_chunks)

        self.frames_group = QtWidgets.QButtonGroup()
        self.radioBtn_allF = QtWidgets.QRadioButton("all frames")
        self.radioBtn_selF = QtWidgets.QRadioButton("active")
        self.frames_group.addButton(self.radioBtn_selF)
        self.frames_group.addButton(self.radioBtn_allF)
        self.radioBtn_allF.setChecked(defaults.all_frames)
        self.radioBtn_selF.setChecked(not defaults.all_frames)

        self.zcxyTxt = QtWidgets.QLabel()
        self.zcxyTxt.setText("Enforce zero cx, cy")
        self.zcxyTxt.setFixedSize(100, 25)

        self.zcxyBox = QtWidgets.QCheckBox()
        self.zcxyBox.setChecked(defaults.zero_cxy)

        self.locFrameTxt = QtWidgets.QLabel()
        self.locFrameTxt.setText("Use localframe")
        self.locFrameTxt.setFixedSize(100, 25)

        self.locFrameBox = QtWidgets.QCheckBox()
        self.locFrameBox.setChecked(defaults.use_localframe)

        self.imgQualTxt = QtWidgets.QLabel()
        self.imgQualTxt.setText("Image quality")
        self.imgQualTxt.setFixedSize(100, 25)

        self.imgQualSpBox = QtWidgets.QSpinBox()
        self.imgQualSpBox.setMinimum(0)
        self.imgQualSpBox.setMaximum(100)
        self.imgQualSpBox.setValue(defaults.image_quality)

        self.expImagesTxt = QtWidgets.QLabel()
        self.expImagesTxt.setText("Export images")
        self.expImagesTxt.setFixedSize(100, 25)

        self.expImagesBox = QtWidgets.QCheckBox()
        self.expImagesBox.setChecked(defaults.export_images)

        self.expMasksTxt = QtWidgets.QLabel()
        self.expMasksTxt.setText("Export masks")
        self.expMasksTxt.setFixedSize(100, 25)

        self.expMasksBox = QtWidgets.QCheckBox()
        self.expMasksBox.setChecked(defaults.export_masks)

        zcxyToolTip = 'Output camera calibrations will have zero cx and cy\nShould be checked until Gaussian Splatting software considers this parameters\nMay result in information loss during export (large cropping)\nTo mitigate that effect, do step 1.1.0. and check "Adaptive camera model fitting" at 1.2. of the script description'
        self.zcxyTxt.setToolTip(zcxyToolTip)
        self.zcxyBox.setToolTip(zcxyToolTip)

        locFrameToolTip = "Shifts coordinates origin to the center of the bounding box\nUses localframe rotation at this point\nThis is useful to fix large coordinates"
        self.locFrameTxt.setToolTip(locFrameToolTip)
        self.locFrameBox.setToolTip(locFrameToolTip)

        imgQualToolTip = "Quality of the output undistorted images (jpeg only)\nMin = 0, Max = 100"
        self.imgQualTxt.setToolTip(imgQualToolTip)
        self.imgQualSpBox.setToolTip(imgQualToolTip)

        expImagesToolTip = "You can disable export of the undistorted images"
        self.expImagesTxt.setToolTip(expImagesToolTip)
        self.expImagesBox.setToolTip(expImagesToolTip)

        expImagesToolTip = "You can enable export of the undistorted masks"
        self.expMasksTxt.setToolTip(expImagesToolTip)
        self.expMasksBox.setToolTip(expImagesToolTip)

        general_layout = QtWidgets.QGridLayout()
        general_layout.setSpacing(9)
        general_layout.addWidget(self.chnkTxt, 1, 0)
        general_layout.addWidget(self.radioBtn_allC, 1, 1)
        general_layout.addWidget(self.radioBtn_selC, 1, 2)
        general_layout.addWidget(self.frmsTxt, 2, 0)
        general_layout.addWidget(self.radioBtn_allF, 2, 1)
        general_layout.addWidget(self.radioBtn_selF, 2, 2)
        general_layout.addWidget(self.zcxyTxt, 3, 0)
        general_layout.addWidget(self.zcxyBox, 3, 1)
        general_layout.addWidget(self.locFrameTxt, 4, 0)
        general_layout.addWidget(self.locFrameBox, 4, 1)
        general_layout.addWidget(self.imgQualTxt, 5, 0)
        general_layout.addWidget(self.imgQualSpBox, 5, 1, 1, 2)

        advanced_layout = QtWidgets.QGridLayout()
        advanced_layout.setSpacing(9)
        advanced_layout.addWidget(self.expImagesTxt, 0, 0)
        advanced_layout.addWidget(self.expImagesBox, 0, 1)
        advanced_layout.setSpacing(9)
        advanced_layout.addWidget(self.expMasksTxt, 1, 0)
        advanced_layout.addWidget(self.expMasksBox, 1, 1)

        self.gbGeneral = QtWidgets.QGroupBox()
        self.gbGeneral.setLayout(general_layout)
        self.gbGeneral.setTitle("General")

        self.gbAdvanced = CollapsibleGroupBox()
        self.gbAdvanced.setLayout(advanced_layout)
        self.gbAdvanced.setTitle("Advanced")
        self.gbAdvanced.setCheckable(True)
        self.gbAdvanced.setChecked(False)
        self.gbAdvanced.toggled.connect(lambda: QtCore.QTimer.singleShot(20, lambda: self.adjustSize()))

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.gbGeneral, 0, 0, 1, 3)
        layout.addWidget(self.gbAdvanced, 1, 0, 1, 3)
        layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding))
        layout.addWidget(self.pBar, 3, 0)
        layout.addWidget(self.btnP1, 3, 1)
        layout.addWidget(self.btnQuit, 3, 2)
        self.setLayout(layout)

        self.buttons = [self.btnP1, self.btnQuit, self.radioBtn_allC, self.radioBtn_selC, self.radioBtn_allF, self.radioBtn_selF, self.zcxyBox, self.locFrameBox, self.imgQualSpBox, self.expImagesBox, self.expMasksBox]

        proc = lambda : self.run_export()

        QtCore.QObject.connect(self.btnP1, QtCore.SIGNAL("clicked()"), proc)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()


def export_for_gaussian_splatting_gui():
    global app
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = ExportSceneGUI(parent)

label = "Scripts/Export Colmap project (for Gaussian Splatting)"
Metashape.app.addMenuItem(label, export_for_gaussian_splatting_gui)
print("To execute this script press {}".format(label))

# If you want to run this script automatically - use this instead:
#export_for_gaussian_splatting()
