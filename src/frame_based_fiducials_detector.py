# How to install:
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
# 2. Restart Metashape
#
# How to use:
# 1. Place fiducials on a few images. We recommend placing on 3 images.
# 2. Run this script from menu Scripts / Detect Fiducials
#
# Script detects scanned aerial image black frame and
# places fiducials relative to this frame
# based on manually marked few images sample.
# This script relies on good frame visibility and
# on constant fiducials position relative to the frame
# which is often not true and results in high errors.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
import math

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def detect_fiducials():
    chunk = Metashape.app.document.chunk
    active_sensors = []

    for sensor in chunk.frame.sensors:
        if not sensor.film_camera:
            continue

        have_enabled_cameras = False
        for cam in chunk.cameras:
            if not cam.enabled or cam.sensor.key != sensor.key:
                continue
            have_enabled_cameras = True
            break

        if not have_enabled_cameras:
            continue

        auto_markers = [None] * 8
        user_markers = []

        for marker in chunk.frame.markers:
            if marker.type == Metashape.Marker.Type.Fiducial and marker.sensor.key == sensor.key:
                if marker.label.startswith("__auto_"):
                    auto_markers[int(marker.label[-1])] = marker
                else:
                    user_markers.append(marker)

        no_projections = False

        for marker in user_markers:
            if len(marker.projections) == 0:
                no_projections = True

        if len(user_markers) == 0 or no_projections:
            raise Exception("For each sensor, mark manually at least 1 image using non-auto fiducials")

        active_sensors.append(sensor)

    chunk.detectFiducials(generate_masks=True, mask_dark_pixels=False, generic_detector=False, frame_detector=True, fiducials_position_corners=False)

    for sensor in active_sensors:
        active_cameras = []
        auto_markers = [None] * 8
        user_markers = []

        for marker in chunk.frame.markers:
            if marker.type == Metashape.Marker.Type.Fiducial and marker.sensor.key == sensor.key:
                if marker.label.startswith("__auto_"):
                    auto_markers[int(marker.label[-1])] = marker
                else:
                    user_markers.append(marker)

        side_markers = auto_markers[4:8]

        sides_valid = all(side_markers)

        if not sides_valid:
            raise Exception("No auto side fiducials found")

        projections = {}
        centers = {}
        rotations = {}
        scales_x = {}
        scales_y = {}
        marker_offsets = {}

        for cam in chunk.cameras:
            if not cam.enabled or cam.sensor.key != sensor.key:
                continue

            side_proj = list(map(lambda el : el.projections[cam], side_markers))

            if not all(side_proj):
                print("Auto fiducials not detected for camera " + cam.label)
                continue

            active_cameras.append(cam)

        for cam in active_cameras:
            side_proj = list(map(lambda el : el.projections[cam], side_markers))

            center = (side_proj[0].coord + side_proj[2].coord + side_proj[1].coord + side_proj[3].coord) / 4
            v1 = side_proj[1].coord - side_proj[3].coord
            v2 = side_proj[2].coord - side_proj[0].coord

            rot = (math.atan2(v1.y, v1.x) + math.atan2(v2.y, v2.x) - math.pi / 2) / 2

            centers[cam] = Metashape.Vector([center[0], center[1]])
            rotations[cam] = rot
            scales_x[cam] = v1.norm();
            scales_y[cam] = v2.norm();

        for user_marker in user_markers:
            offs = Metashape.Vector([0, 0])
            nsamples = 0

            for cam in active_cameras:
                if not user_marker.projections[cam]:
                    continue

                rot = -rotations[cam]

                user_coord = user_marker.projections[cam].coord
                user_coord = Metashape.Vector([user_coord[0], user_coord[1]])
                user_coord = user_coord - centers[cam]
                user_coord = Metashape.Vector([user_coord.x * math.cos(rot) - user_coord.y * math.sin(rot), user_coord.x * math.sin(rot) + user_coord.y * math.cos(rot)])
                user_coord.x /= scales_x[cam]
                user_coord.y /= scales_y[cam]

                offs += user_coord
                nsamples += 1

            if nsamples == 0:
                raise Error("Specify at least 1 projection for fiducial " + user_marker.label);

            offs /= nsamples
            marker_offsets[user_marker] = offs


        for user_marker in user_markers:
            offs = marker_offsets[user_marker]

            for cam in active_cameras:
                if user_marker.projections[cam] and user_marker.projections[cam].valid:
                    continue

                rot = rotations[cam]

                user_coord = Metashape.Vector([offs.x, offs.y])
                user_coord.x *= scales_x[cam]
                user_coord.y *= scales_y[cam]
                user_coord = Metashape.Vector([user_coord.x * math.cos(rot) - user_coord.y * math.sin(rot), user_coord.x * math.sin(rot) + user_coord.y * math.cos(rot)])

                user_coord += centers[cam]

                user_marker.projections[cam] = Metashape.Marker.Projection()
                user_marker.projections[cam].coord = user_coord
                user_marker.projections[cam].valid = True

        for auto_marker in auto_markers:
            if(not auto_marker):
                continue
            chunk.remove(auto_marker)

label = "Scripts/Detect Fiducials"
Metashape.app.addMenuItem(label, detect_fiducials)
print("To execute this script press {}".format(label))
