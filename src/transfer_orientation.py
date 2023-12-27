# Transfers orientations from certain cameras to the corresponding ones.
#
# Usecase: RGB and thermal photos were taken simultaneously and in the same directions.
# Alignment can be computed only with RGB photos and then transferred to thermal.
#
# Important: Calibration for thermal cameras will not be adjusted automatically.
#
# Usage:
# 1. Chunk (right click) -> Add -> Add Folder... to add RGB photos
# 2. Chunk (right click) -> Add -> Add Photos... to add thermal photos (using "Multi-camera system" option)
# 3. Disable all thermal cameras (for example, in the Photos pane)
# 4. Workflow -> Align Photos... (only RGB photos will be aligned)
# 5. Enable all cameras and click Scripts -> Transfer orientations
#

import Metashape
import datetime as dt
from datetime import datetime, timedelta

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def check_camera_master(cam):
    return cam.master == cam


def check_camera_transform(cam):
    return cam.transform is not None


def get_number(name):
    numbers = []
    cur_number = ""
    for c in name:
        if (c in "0123456789"):
            cur_number = cur_number + c
        else:
            if (cur_number != ""):
                numbers.append(cur_number)
                cur_number = ""
    if (cur_number != ""):
        numbers.append(cur_number)
    number = ""
    for n in numbers:
        if (len(n) >= len(number)):
            number = n
    return (int(number) if number != "" else 0)


def parse_datetime(time):
    try:
        return datetime.strptime(time, "%Y:%m:%d %H:%M:%S")
    except:
        return datetime(dt.MINYEAR, 1, 1)


def get_camera_meta(cam):
    meta = cam.photo.meta
    res = [cam]
    res.append(get_number(cam.label))
    res.append(parse_datetime(meta["Exif/DateTime"]))
    return res


def find_correspondence(cams_0, cams_1):
    links_0 = [[] for c in cams_0]
    links_1 = [[] for c in cams_1]
    shift_stats = {}

    pos_0 = 0
    for pos_1 in range(len(cams_1)):
        t = cams_1[pos_1][2]
        t_margin = timedelta(seconds=1)
        t_lower = t - t_margin
        t_upper = t + t_margin
        while (pos_0 > 0 and cams_0[pos_0 - 1][2] >= t_lower):
            pos_0 -= 1
        while (pos_0 < len(cams_0) and cams_0[pos_0][2] <= t_upper):
            if (cams_0[pos_0][2] >= t_lower):
                links_1[pos_1].append(pos_0)
                links_0[pos_0].append(pos_1)

                shift = cams_0[pos_0][1] - cams_1[pos_1][1]
                dt = (cams_0[pos_0][2] - cams_1[pos_1][2]).total_seconds()
                stat = shift_stats.get(shift, (0, 0))
                shift_stats[shift] = (stat[0] + abs(dt), stat[1] + 1)
            pos_0 += 1

    shift_stats = {shift: (shift_stats[shift][0] / shift_stats[shift][1], -shift_stats[shift][1]) for shift in shift_stats}
    shifts = sorted(shift_stats.keys(), key=lambda shift: shift_stats[shift])

    res_0 = [None for c in cams_0]
    res_1 = [None for c in cams_1]
    unpaired = []
    unpaired_next = list(range(len(cams_1)))

    for shift in shifts:
        unpaired = unpaired_next
        unpaired_next = []

        if (len(unpaired) == 0):
            break

        for pos_1 in unpaired:
            best_pos_0 = None
            best_dt = 0
            for pos_0 in links_1[pos_1]:
                cur_shift = cams_0[pos_0][1] - cams_1[pos_1][1]
                if (cur_shift == shift and res_0[pos_0] is None):
                    cur_dt = abs((cams_0[pos_0][2] - cams_1[pos_1][2]).total_seconds())
                    if (best_pos_0 is None or cur_dt < best_dt):
                        best_pos_0 = pos_0
                        best_dt = cur_dt
            if (best_pos_0 is None):
                unpaired_next.append(pos_1)
            else:
                res_0[best_pos_0] = pos_1
                res_1[pos_1] = best_pos_0

    return res_1


def transfer_orientations():
    chunk = Metashape.app.document.chunk

    enabled_cameras = list(filter(lambda c: ((c.type == Metashape.Camera.Type.Regular) and c.enabled), chunk.cameras))
    master_cameras = list(filter(check_camera_master, enabled_cameras))

    cameras_estimated = list(filter(check_camera_transform, master_cameras))
    cameras_not_estimated = list(filter(lambda c: not check_camera_transform(c), master_cameras))

    cameras_estimated = [get_camera_meta(c) for c in cameras_estimated]
    cameras_not_estimated = [get_camera_meta(c) for c in cameras_not_estimated]

    cameras_estimated.sort(key=lambda c: c[2])
    cameras_not_estimated.sort(key=lambda c: c[2])

    correspondence = find_correspondence(cameras_estimated, cameras_not_estimated)

    transferred_cnt = 0
    unmatched = []
    for pos_1, cam_1 in enumerate(cameras_not_estimated):
        pos_0 = correspondence[pos_1]
        if (pos_0 is None):
            unmatched.append(cam_1[0])
            continue
        cam_0 = cameras_estimated[pos_0]
        cam_1[0].transform = cam_0[0].transform
        transferred_cnt += 1

    print("------------------")
    if (transferred_cnt > 0):
        print("Successfully transferred {} orientations".format(transferred_cnt))
    else:
        print("Transferred {} orientations".format(transferred_cnt))
    if (len(unmatched) > 0):
        print("{} cameras remain without orientations:".format(len(unmatched)))
        for cam in unmatched:
            print(cam)


label = "Scripts/Transfer orientations"
Metashape.app.addMenuItem(label, transfer_orientations)
print("To execute this script press {}".format(label))
