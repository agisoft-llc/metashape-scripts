# Builds physically based rendering (PBR) textures from a set of aligned
# photos and a UV-mapped mesh: albedo, roughness, metallic and normal maps.
#
# The model's world-space positions, normals and tangents are rasterized into
# the mesh's UV space on the CPU. An OpenCL pipeline then gathers per-texel
# "witnesses" (the photo sample, view/light directions and visibility for every
# camera that sees the texel) and optimizes the PBR maps against them. An
# optional light fit first calibrates the flash multiplier and camera-space
# offset using low-resolution discovery-only test builds.
#
# An OpenCL device is required; numpy is installed on first run.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import contextlib
import ctypes
import ctypes.util
import glob
import importlib
import json
import os
import subprocess
import sys
import time
import traceback

import Metashape
from PySide2 import QtCore, QtWidgets

compatible_major_version = "2.3"
found_major_version = ".".join(Metashape.app.version.split(".")[:2])
if found_major_version != compatible_major_version:
    raise Exception(
        f"Incompatible Metashape version: {found_major_version} != {compatible_major_version}"
    )

DEFAULT_BAKE_SIZE = 1024
MAX_BAKE_SIZE = 8192

MAPPING_MODES = (
    ("Generic", Metashape.MappingMode.GenericMapping),
    ("Orthophoto", Metashape.MappingMode.OrthophotoMapping),
    ("Adaptive orthophoto", Metashape.MappingMode.AdaptiveOrthophotoMapping),
    ("Spherical", Metashape.MappingMode.SphericalMapping),
    ("Camera", Metashape.MappingMode.CameraMapping),
)

CL_SUCCESS = 0
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_PLATFORM_NAME = 0x0902
CL_DEVICE_NAME = 0x102B
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_READ_WRITE = 1 << 0
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_PROGRAM_BUILD_LOG = 0x1183

_CL_POINTER = ctypes.c_void_p
_CL_SIZE = ctypes.c_size_t
_CL_UINT = ctypes.c_uint
_CL_INT = ctypes.c_int
_CL_FLAGS = ctypes.c_uint64

_OPENCL = None


class _Cancelled(Exception):
    """Raised by the pipeline when the progress dialog is cancelled."""


class _CancelToken:
    """One-way cancellation flag shared by the dialog and the pipeline."""

    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def check(self):
        if self.cancelled:
            raise _Cancelled()


def _check_cancel(cancel):
    if cancel is not None:
        cancel.check()


def _user_packages_path():
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.DataLocation
    )
    tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return os.path.join(location, "user-packages-" + tag)


def _ensure_package(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

    target = _user_packages_path()
    os.makedirs(target, exist_ok=True)
    site_path = os.path.join(
        target,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    if site_path not in sys.path:
        sys.path.append(site_path)

    print(f"Installing script dependency: {name}...")
    env = dict(os.environ, PYTHONUSERBASE=target)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "--no-warn-script-location",
            name,
        ],
        env=env,
    )
    importlib.invalidate_caches()
    return importlib.import_module(name)


np = _ensure_package("numpy")


def _load_opencl():
    candidates = []
    library = ctypes.util.find_library("OpenCL")
    if library:
        candidates.append(library)
    candidates += ["OpenCL.dll", "libOpenCL.so.1", "libOpenCL.so"]
    candidates += glob.glob("/nix/store/*ocl-icd*/lib/libOpenCL.so*")
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise RuntimeError("OpenCL is not available (OpenCL.dll / libOpenCL.so not found)")


def _opencl():
    global _OPENCL
    if _OPENCL is None:
        _OPENCL = _load_opencl()
        _OPENCL.clGetPlatformIDs.argtypes = [
            _CL_UINT,
            ctypes.POINTER(_CL_POINTER),
            ctypes.POINTER(_CL_UINT),
        ]
        _OPENCL.clGetPlatformInfo.argtypes = [
            _CL_POINTER,
            _CL_UINT,
            _CL_SIZE,
            _CL_POINTER,
            ctypes.POINTER(_CL_SIZE),
        ]
        _OPENCL.clGetDeviceIDs.argtypes = [
            _CL_POINTER,
            _CL_FLAGS,
            _CL_UINT,
            ctypes.POINTER(_CL_POINTER),
            ctypes.POINTER(_CL_UINT),
        ]
        _OPENCL.clGetDeviceInfo.argtypes = [
            _CL_POINTER,
            _CL_UINT,
            _CL_SIZE,
            _CL_POINTER,
            ctypes.POINTER(_CL_SIZE),
        ]
        _OPENCL.clCreateContext.restype = _CL_POINTER
        _OPENCL.clCreateContext.argtypes = [
            _CL_POINTER,
            _CL_UINT,
            ctypes.POINTER(_CL_POINTER),
            _CL_POINTER,
            _CL_POINTER,
            ctypes.POINTER(_CL_INT),
        ]
        _OPENCL.clCreateCommandQueue.restype = _CL_POINTER
        _OPENCL.clCreateCommandQueue.argtypes = [
            _CL_POINTER,
            _CL_POINTER,
            _CL_FLAGS,
            ctypes.POINTER(_CL_INT),
        ]
        _OPENCL.clCreateBuffer.restype = _CL_POINTER
        _OPENCL.clCreateBuffer.argtypes = [
            _CL_POINTER,
            _CL_FLAGS,
            _CL_SIZE,
            _CL_POINTER,
            ctypes.POINTER(_CL_INT),
        ]
        _OPENCL.clCreateProgramWithSource.restype = _CL_POINTER
        _OPENCL.clCreateProgramWithSource.argtypes = [
            _CL_POINTER,
            _CL_UINT,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(_CL_SIZE),
            ctypes.POINTER(_CL_INT),
        ]
        _OPENCL.clBuildProgram.argtypes = [
            _CL_POINTER,
            _CL_UINT,
            ctypes.POINTER(_CL_POINTER),
            ctypes.c_char_p,
            _CL_POINTER,
            _CL_POINTER,
        ]
        _OPENCL.clGetProgramBuildInfo.argtypes = [
            _CL_POINTER,
            _CL_POINTER,
            _CL_UINT,
            _CL_SIZE,
            _CL_POINTER,
            ctypes.POINTER(_CL_SIZE),
        ]
        _OPENCL.clCreateKernel.restype = _CL_POINTER
        _OPENCL.clCreateKernel.argtypes = [
            _CL_POINTER,
            ctypes.c_char_p,
            ctypes.POINTER(_CL_INT),
        ]
        _OPENCL.clSetKernelArg.argtypes = [_CL_POINTER, _CL_UINT, _CL_SIZE, _CL_POINTER]
        _OPENCL.clEnqueueNDRangeKernel.argtypes = [
            _CL_POINTER,
            _CL_POINTER,
            _CL_UINT,
            ctypes.POINTER(_CL_SIZE),
            ctypes.POINTER(_CL_SIZE),
            ctypes.POINTER(_CL_SIZE),
            _CL_UINT,
            _CL_POINTER,
            _CL_POINTER,
        ]
        _OPENCL.clEnqueueReadBuffer.argtypes = [
            _CL_POINTER,
            _CL_POINTER,
            _CL_UINT,
            _CL_SIZE,
            _CL_SIZE,
            _CL_POINTER,
            _CL_UINT,
            _CL_POINTER,
            _CL_POINTER,
        ]
        _OPENCL.clEnqueueWriteBuffer.argtypes = [
            _CL_POINTER,
            _CL_POINTER,
            _CL_UINT,
            _CL_SIZE,
            _CL_SIZE,
            _CL_POINTER,
            _CL_UINT,
            _CL_POINTER,
            _CL_POINTER,
        ]
        _OPENCL.clFinish.argtypes = [_CL_POINTER]
        _OPENCL.clReleaseMemObject.argtypes = [_CL_POINTER]
        _OPENCL.clReleaseKernel.argtypes = [_CL_POINTER]
        _OPENCL.clReleaseProgram.argtypes = [_CL_POINTER]
        _OPENCL.clReleaseCommandQueue.argtypes = [_CL_POINTER]
        _OPENCL.clReleaseContext.argtypes = [_CL_POINTER]
    return _OPENCL


def _check(result, operation):
    if result != CL_SUCCESS:
        raise RuntimeError(f"{operation} failed with OpenCL error {result}")


_CL_SHARED = {}


def _shared_cl(device):
    key = int(ctypes.cast(device, _CL_POINTER).value)
    if key not in _CL_SHARED:
        cl = _opencl()
        error = _CL_INT(0)
        context = cl.clCreateContext(
            None, 1, (_CL_POINTER * 1)(device), None, None, ctypes.byref(error)
        )
        _check(error.value, "clCreateContext")
        queue = cl.clCreateCommandQueue(context, device, 0, ctypes.byref(error))
        _check(error.value, "clCreateCommandQueue")
        _CL_SHARED[key] = (context, queue)
    return _CL_SHARED[key]


def _build_program(cl, context, device, source, kernel_name):
    """Compiles source and returns (program, kernel); raises with the build log."""
    error = _CL_INT(0)
    strings = (ctypes.c_char_p * 1)(source)
    lengths = (_CL_SIZE * 1)(len(source))
    program = cl.clCreateProgramWithSource(
        context, 1, strings, lengths, ctypes.byref(error)
    )
    _check(error.value, "clCreateProgramWithSource")
    result = cl.clBuildProgram(program, 1, (_CL_POINTER * 1)(device), None, None, None)
    if result != CL_SUCCESS:
        size = _CL_SIZE(0)
        cl.clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(size)
        )
        log = ctypes.create_string_buffer(size.value)
        cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, None)
        raise RuntimeError(
            f"OpenCL kernel '{kernel_name}' failed to build ({result}):\n{log.value.decode()}"
        )
    kernel = cl.clCreateKernel(program, kernel_name.encode(), ctypes.byref(error))
    _check(error.value, "clCreateKernel")
    return program, kernel


def _create_buffer(cl, context, size, flags, data=None):
    error = _CL_INT(0)
    host = None
    if data is not None:
        host = ctypes.create_string_buffer(data, len(data))
        flags |= CL_MEM_COPY_HOST_PTR
    memory = cl.clCreateBuffer(
        context,
        flags,
        size,
        ctypes.cast(host, _CL_POINTER) if host is not None else None,
        ctypes.byref(error),
    )
    _check(error.value, "clCreateBuffer")
    return memory


def _write_buffer(cl, queue, memory, data):
    host = ctypes.create_string_buffer(data, len(data))
    cl.clEnqueueWriteBuffer(
        queue, memory, 1, 0, len(data), ctypes.cast(host, _CL_POINTER), 0, None, None
    )


def _read_buffer(cl, queue, memory, size, dtype):
    output = (ctypes.c_char * size)()
    cl.clEnqueueReadBuffer(
        queue, memory, 1, 0, size, ctypes.cast(output, _CL_POINTER), 0, None, None
    )
    cl.clFinish(queue)
    return np.frombuffer(bytes(output), dtype=dtype)


class _Kernel:
    """An OpenCL kernel with typed argument setters and a launch wrapper."""

    def __init__(self, cl, queue, kernel):
        self._cl = cl
        self._queue = queue
        self._kernel = kernel

    @property
    def handle(self):
        return self._kernel

    def set_buffer(self, index, memory):
        self._cl.clSetKernelArg(
            self._kernel,
            index,
            ctypes.sizeof(_CL_POINTER),
            ctypes.byref(_CL_POINTER(memory)),
        )

    def set_uint(self, index, value):
        self._cl.clSetKernelArg(
            self._kernel,
            index,
            ctypes.sizeof(_CL_UINT),
            ctypes.byref(_CL_UINT(int(value))),
        )

    def set_int(self, index, value):
        self._cl.clSetKernelArg(
            self._kernel,
            index,
            ctypes.sizeof(_CL_INT),
            ctypes.byref(_CL_INT(int(value))),
        )

    def set_float(self, index, value):
        self._cl.clSetKernelArg(
            self._kernel,
            index,
            ctypes.sizeof(ctypes.c_float),
            ctypes.byref(ctypes.c_float(value)),
        )

    def launch(self, global_size, local_size=64):
        global_work = (_CL_SIZE * 1)(global_size)
        local_work = (_CL_SIZE * 1)(local_size) if local_size else None
        _check(
            self._cl.clEnqueueNDRangeKernel(
                self._queue,
                self._kernel,
                1,
                None,
                global_work,
                local_work,
                0,
                None,
                None,
            ),
            "clEnqueueNDRangeKernel",
        )
        self._cl.clFinish(self._queue)


class _OpenCLProgram:
    """Shared context/queue and buffer helpers for the GPU pipeline classes."""

    def __init__(self, device):
        self._cl = _opencl()
        self._context, self._queue = _shared_cl(device)

    def _buffer(self, size, flags, data=None):
        return _create_buffer(self._cl, self._context, size, flags, data)

    def _write(self, memory, data):
        _write_buffer(self._cl, self._queue, memory, data)

    def _read(self, memory, size, dtype):
        return _read_buffer(self._cl, self._queue, memory, size, dtype)


def _cl_info(getter, handle, parameter):
    size = _CL_SIZE(0)
    getter(handle, parameter, 0, None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    getter(handle, parameter, size, buffer, None)
    return buffer.value.decode()


def _opencl_devices():
    cl = _opencl()
    platform_count = _CL_UINT(0)
    cl.clGetPlatformIDs(0, None, ctypes.byref(platform_count))
    if platform_count.value == 0:
        return []
    platforms = (_CL_POINTER * platform_count.value)()
    cl.clGetPlatformIDs(platform_count, platforms, None)

    devices = []
    for platform_index in range(platform_count.value):
        platform = platforms[platform_index]
        platform_name = _cl_info(cl.clGetPlatformInfo, platform, CL_PLATFORM_NAME)
        device_count = _CL_UINT(0)
        if (
            cl.clGetDeviceIDs(
                platform, CL_DEVICE_TYPE_ALL, 0, None, ctypes.byref(device_count)
            )
            != CL_SUCCESS
        ):
            continue
        if device_count.value == 0:
            continue
        handles = (_CL_POINTER * device_count.value)()
        cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count, handles, None)
        for device_index in range(device_count.value):
            device_name = _cl_info(
                cl.clGetDeviceInfo, handles[device_index], CL_DEVICE_NAME
            )
            devices.append((handles[device_index], f"{platform_name}: {device_name}"))
    return devices


def _mesh_arrays(model):
    vertices = np.empty((len(model.vertices), 3), np.float32)
    for i, vertex in enumerate(model.vertices):
        coord = vertex.coord
        vertices[i] = (coord[0], coord[1], coord[2])

    transform = model.transform
    if transform is not None:
        matrix = np.array(
            [[transform[r, c] for c in range(4)] for r in range(4)], np.float64
        )
        vertices = (
            np.concatenate([vertices, np.ones((len(vertices), 1), np.float32)], axis=1)
            @ matrix.T
        )[:, :3].astype(np.float32)

    face_count = len(model.faces)
    faces = np.empty((face_count, 3), np.int32)
    tex_indices = np.empty((face_count, 3), np.int32)
    for i, face in enumerate(model.faces):
        indices = face.vertices
        texels = face.tex_vertices
        faces[i] = (indices[0], indices[1], indices[2])
        tex_indices[i] = (texels[0], texels[1], texels[2])

    tex_coords = np.empty((len(model.tex_vertices), 2), np.float32)
    for i, vertex in enumerate(model.tex_vertices):
        coord = vertex.coord
        tex_coords[i] = (coord[0], 1.0 - coord[1])

    return vertices, faces, tex_indices, tex_coords


def _vertex_normals(vertices, faces):
    corners = vertices[faces]
    face_normals = np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )

    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths


def _vertex_tangents(vertices, faces, tex_indices, tex_coords):
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    uv0 = tex_coords[tex_indices[:, 0]]
    uv1 = tex_coords[tex_indices[:, 1]]
    uv2 = tex_coords[tex_indices[:, 2]]

    edge1 = p1 - p0
    edge2 = p2 - p0
    delta1 = uv1 - uv0
    delta2 = uv2 - uv0
    determinant = delta1[:, 0] * delta2[:, 1] - delta2[:, 0] * delta1[:, 1]

    face_tangents = edge1 * delta2[:, 1:2] - edge2 * delta1[:, 1:2]
    valid = np.abs(determinant) > 1e-6
    face_tangents[valid] /= determinant[valid, None]
    face_tangents[~valid] = (1.0, 0.0, 0.0)

    tangents = np.zeros((len(tex_coords), 3), np.float32)
    for corner in range(3):
        np.add.at(tangents, tex_indices[:, corner], face_tangents)

    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return tangents / lengths


def _shift(array, dy, dx):
    pad = ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0))) + ((0, 0),) * (
        array.ndim - 2
    )
    padded = np.pad(array, pad)
    return padded[
        max(-dy, 0) : max(-dy, 0) + array.shape[0],
        max(-dx, 0) : max(-dx, 0) + array.shape[1],
    ]


def _dilate(image, iterations=10):
    filled = ~np.isnan(image[..., 0])
    result = image.copy()
    for _ in range(iterations):
        if filled.all():
            break
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            source = _shift(result, dy, dx)
            source_filled = _shift(filled, dy, dx)
            fill = (~filled) & source_filled
            result[fill] = source[fill]
            filled |= fill
    return result


def _bake_geometry(model, size, report=None, cancel=None):
    vertices, faces, tex_indices, tex_coords = _mesh_arrays(model)
    normals = _vertex_normals(vertices, faces)
    valid = np.all(tex_indices >= 0, axis=1)
    tangents = _vertex_tangents(vertices, faces[valid], tex_indices[valid], tex_coords)

    tri_uv = tex_coords[tex_indices[valid]] * size
    tri_positions = vertices[faces[valid]]
    tri_normals = normals[faces[valid]]
    tri_tangents = tangents[tex_indices[valid]]

    positions = np.full((size, size, 3), np.nan, np.float32)
    normal_map = np.full((size, size, 3), np.nan, np.float32)
    tangent_map = np.full((size, size, 3), np.nan, np.float32)

    x0, y0 = tri_uv[:, 0, 0], tri_uv[:, 0, 1]
    x1, y1 = tri_uv[:, 1, 0], tri_uv[:, 1, 1]
    x2, y2 = tri_uv[:, 2, 0], tri_uv[:, 2, 1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    triangle_count = len(tri_uv)
    report_step = max(1, triangle_count // 100)
    for triangle in range(triangle_count):
        if triangle % report_step == 0:
            _check_cancel(cancel)
        if report is not None and triangle % report_step == 0:
            report(
                triangle / max(1, triangle_count),
                f"Baking geometry maps: {int(100 * triangle / max(1, triangle_count))}%",
            )
        double_area = area[triangle]
        if abs(double_area) < 1e-12:
            continue

        min_x = max(
            int(np.floor(min(x0[triangle], x1[triangle], x2[triangle]) - 0.5)), 0
        )
        max_x = min(
            int(np.ceil(max(x0[triangle], x1[triangle], x2[triangle]) - 0.5)), size - 1
        )
        min_y = max(
            int(np.floor(min(y0[triangle], y1[triangle], y2[triangle]) - 0.5)), 0
        )
        max_y = min(
            int(np.ceil(max(y0[triangle], y1[triangle], y2[triangle]) - 0.5)), size - 1
        )
        if min_x > max_x or min_y > max_y:
            continue

        rows, cols = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        px = cols + 0.5
        py = rows + 0.5
        w0 = (
            (x1[triangle] - px) * (y2[triangle] - py)
            - (x2[triangle] - px) * (y1[triangle] - py)
        ) / double_area
        w1 = (
            (x2[triangle] - px) * (y0[triangle] - py)
            - (x0[triangle] - px) * (y2[triangle] - py)
        ) / double_area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue

        selected_rows = rows[inside]
        selected_cols = cols[inside]
        weights = np.stack([w0[inside], w1[inside], w2[inside]], axis=-1)

        positions[selected_rows, selected_cols] = weights @ tri_positions[triangle]
        normal = weights @ tri_normals[triangle]
        lengths = np.linalg.norm(normal, axis=-1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        normal_map[selected_rows, selected_cols] = normal / lengths
        tangent = weights @ tri_tangents[triangle]
        lengths = np.linalg.norm(tangent, axis=-1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        tangent_map[selected_rows, selected_cols] = tangent / lengths

    return _dilate(positions), _dilate(normal_map), _dilate(tangent_map)


def _save_array_png(path, image):
    data = (np.clip(np.nan_to_num(image), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Metashape.Image.fromstring(
        data.tobytes(), image.shape[1], image.shape[0], "RGB", "U8"
    ).save(path)


def _save_baked_maps(out_dir, positions, normals, tangents):
    baked_dir = os.path.join(out_dir, "baked")
    os.makedirs(baked_dir, exist_ok=True)
    finite = positions[~np.isnan(positions[..., 0])]
    if len(finite):
        low = finite.min(axis=0)
        span = finite.max(axis=0) - low
        span[span == 0.0] = 1.0
        positions = (positions - low) / span
    _save_array_png(os.path.join(baked_dir, "position.png"), positions)
    _save_array_png(os.path.join(baked_dir, "normal.png"), normals * 0.5 + 0.5)
    _save_array_png(os.path.join(baked_dir, "tangent.png"), tangents * 0.5 + 0.5)
    print("Baked maps saved to: " + baked_dir)


GATHER_DEPTH_EPS_ABS = 0.05
GATHER_DEPTH_EPS_REL = 0.005
GATHER_OCCLUSION_RADIUS = 16
GATHER_OCCLUSION_NEAR_FADE = 1.0
GATHER_OCCLUSION_VALID_FADE = 1.0
GATHER_SHADING_GATE = 0.05
WITNESS_WORDS = 3

INITIAL_ALBEDO = 0.75
INITIAL_ROUGHNESS = 0.1
INITIAL_METALLIC = 0.1
INITIAL_NORMAL = (0.5, 0.5, 1.0)


def _camera_axes(camera):
    transform = camera.transform
    matrix = np.array(
        [[transform[r, c] for c in range(4)] for r in range(4)], np.float64
    )
    return matrix[:3, 3], matrix[:3, 0], -matrix[:3, 1], matrix[:3, 2]


PROJECTION_LUT_GRID = 128
PROJECTION_LUT_MARGIN = 1.5


def _projection_lut(camera):
    calibration = camera.sensor.calibration
    cam_w, cam_h = int(calibration.width), int(calibration.height)
    half_s = (cam_w * 0.5) / float(calibration.f)
    half_t = (cam_h * 0.5) / float(calibration.f)
    s_min = -half_s * PROJECTION_LUT_MARGIN
    s_span = 2.0 * half_s * PROJECTION_LUT_MARGIN
    t_min = -half_t * PROJECTION_LUT_MARGIN
    t_span = 2.0 * half_t * PROJECTION_LUT_MARGIN
    grid = PROJECTION_LUT_GRID
    lut = np.empty((grid * grid * 2,), np.float32)
    transform = camera.transform
    for j in range(grid):
        t = t_min + t_span * j / (grid - 1)
        for i in range(grid):
            s = s_min + s_span * i / (grid - 1)
            world = transform.mulp(Metashape.Vector([s, t, 1.0]))
            uv = camera.project(world)
            index = (j * grid + i) * 2
            if uv is None:
                lut[index] = -1.0
                lut[index + 1] = -1.0
            else:
                lut[index] = uv.x
                lut[index + 1] = uv.y
    return lut, grid, s_min, s_span, t_min, t_span


_COMMON_HELPERS = b"""
static inline float smoothstep01(float t) {
  t = clamp(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

// Decodes a tangent-space normal sample into world space.
static inline float3 getNormal(float3 geom_normal, float3 normal_sample, float3 tangent) {
  float3 bitangent = normalize(cross(geom_normal, tangent));
  float3 u = normalize(normal_sample * 2.0f - (float3)(1.0f));
  return normalize(tangent * u.x + bitangent * u.y + geom_normal * u.z);
}

static inline uint floatToHalf(float f) {
  uint bits = as_uint(f);
  uint sign = (bits >> 16) & 0x8000u;
  int exponent = (int)((bits >> 23) & 0xFFu) - 127 + 15;
  uint mantissa = bits & 0x7FFFFFu;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    uint sub = (mantissa | 0x800000u) >> (1 - exponent);
    return sign | (sub >> 13);
  }
  if (exponent >= 31) return sign | 0x7C00u;
  return sign | ((uint)exponent << 10) | (mantissa >> 13);
}

static inline float halfToFloat(uint h) {
  uint sign = (h & 0x8000u) << 16;
  uint exp = (h >> 10) & 0x1Fu;
  uint mant = h & 0x3FFu;
  uint bits;
  if (exp == 0u) {
    bits = sign;
  } else if (exp == 31u) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  }
  return as_float(bits);
}
"""

GATHER_KERNEL = (
    _COMMON_HELPERS
    + b"""
static inline float2 sample2(__global const float *image, uint width, uint height, float x, float y) {
  x = clamp(x, 0.0f, (float)(width - 1) - 0.001f);
  y = clamp(y, 0.0f, (float)(height - 1) - 0.001f);
  int x0 = (int)floor(x), y0 = (int)floor(y);
  int x1 = min(x0 + 1, (int)width - 1), y1 = min(y0 + 1, (int)height - 1);
  float fx = x - x0, fy = y - y0;
  uint i00 = (uint)((y0 * width + x0) * 2), i10 = (uint)((y0 * width + x1) * 2);
  uint i01 = (uint)((y1 * width + x0) * 2), i11 = (uint)((y1 * width + x1) * 2);
  float2 a = (float2)(image[i00], image[i00 + 1]);
  float2 b = (float2)(image[i10], image[i10 + 1]);
  float2 c = (float2)(image[i01], image[i01 + 1]);
  float2 d = (float2)(image[i11], image[i11 + 1]);
  return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

static inline float3 sample3(__global const float *image, uint width, uint height, float x, float y) {
  x = clamp(x, 0.0f, (float)(width - 1) - 0.001f);
  y = clamp(y, 0.0f, (float)(height - 1) - 0.001f);
  int x0 = (int)floor(x), y0 = (int)floor(y);
  int x1 = min(x0 + 1, (int)width - 1), y1 = min(y0 + 1, (int)height - 1);
  float fx = x - x0, fy = y - y0;
  uint i00 = (uint)((y0 * width + x0) * 3), i10 = (uint)((y0 * width + x1) * 3);
  uint i01 = (uint)((y1 * width + x0) * 3), i11 = (uint)((y1 * width + x1) * 3);
  float3 a = (float3)(image[i00], image[i00 + 1], image[i00 + 2]);
  float3 b = (float3)(image[i10], image[i10 + 1], image[i10 + 2]);
  float3 c = (float3)(image[i01], image[i01 + 1], image[i01 + 2]);
  float3 d = (float3)(image[i11], image[i11 + 1], image[i11 + 2]);
  return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

__kernel void gather(
    __global const float *positions,
    __global const float *normals,
    __global const float *lut,
    __global const float *depth_image,
    __global const float *photo,
    __global float *weights,
    __global uint *pool,
    __global uint *counts,
    __global const float *params,
    const uint texel_count,
    const uint cap,
    const uint lut_grid,
    const uint cam_w,
    const uint cam_h,
    const uint photo_w,
    const uint photo_h,
    const int occlusion_radius,
    const uint camera_index)
{
  uint i = get_global_id(0);
  if (i >= texel_count) return;
  float3 p = (float3)(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
  if (isnan(p.x)) return;

  float3 cam_pos = (float3)(params[0], params[1], params[2]);
  float3 right = (float3)(params[3], params[4], params[5]);
  float3 down = (float3)(params[6], params[7], params[8]);
  float3 forward = (float3)(params[9], params[10], params[11]);
  float3 light_pos = (float3)(params[12], params[13], params[14]);
  float lut_smin = params[16], lut_sspan = params[17];
  float lut_tmin = params[18], lut_tspan = params[19];
  float depth_eps_abs = params[20], depth_eps_rel = params[21];
  float shading_gate = params[22], border_fade = params[23];
  float occlusion_near_fade = params[24], occlusion_valid_fade = params[25];

  float3 delta = p - cam_pos;
  float cam_z = dot(delta, forward);
  if (cam_z <= 1e-6f) return;
  float s = dot(delta, right) / cam_z;
  float t = dot(delta, down) / cam_z;
  float gx = (s - lut_smin) / lut_sspan * (float)(lut_grid - 1);
  float gy = (t - lut_tmin) / lut_tspan * (float)(lut_grid - 1);
  if (gx < 0.0f || gy < 0.0f || gx > (float)(lut_grid - 1) || gy > (float)(lut_grid - 1)) return;

  float2 uv = sample2(lut, lut_grid, lut_grid, gx, gy);
  if (uv.x < 0.0f || uv.y < 0.0f || uv.x >= (float)cam_w || uv.y >= (float)cam_h) return;

  int ipx = (int)floor(uv.x), ipy = (int)floor(uv.y);
  int valid_count = 0, near_count = 0, total = 0;
  for (int dy = -occlusion_radius; dy <= occlusion_radius; ++dy) {
    for (int dx = -occlusion_radius; dx <= occlusion_radius; ++dx) {
      int sx = ipx + dx, sy = ipy + dy;
      if (sx < 0 || sy < 0 || sx >= (int)cam_w || sy >= (int)cam_h) continue;
      ++total;
      uint di = (uint)((sy * cam_w + sx) * 2);
      float2 sample = (float2)(depth_image[di], depth_image[di + 1]);
      if (isnan(sample.x) || sample.y < 0.5f) continue;
      ++valid_count;
      if (sample.x < cam_z - (depth_eps_abs + depth_eps_rel * fabs(cam_z))) ++near_count;
    }
  }
  if (valid_count == 0) return;
  float valid_fraction = (float)valid_count / (float)total;
  float near_fraction = (float)near_count / (float)valid_count;
  float occlusion = smoothstep01(valid_fraction / occlusion_valid_fade) *
                    (1.0f - smoothstep01(near_fraction / occlusion_near_fade));
  if (occlusion < 1e-4f) return;

  float border = fmin(fmin(uv.x, (float)cam_w - 1.0f - uv.x),
                      fmin(uv.y, (float)cam_h - 1.0f - uv.y));
  float photo_gate = smoothstep01(border / border_fade);
  if (photo_gate < 1e-4f) return;

  float3 sample = sample3(photo, photo_w, photo_h, uv.x, uv.y);
  float3 n = (float3)(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
  float3 view_dir = normalize(cam_pos - p);
  float3 light_dir = normalize(light_pos - p);
  float NdotL = fmax(0.0f, dot(n, light_dir));
  float NdotV = fmax(0.0f, dot(n, view_dir));
  float shading = smoothstep01(NdotL / shading_gate) * smoothstep01(NdotV / shading_gate);
  if (shading * photo_gate < 1e-4f) return;

  float3 delta_cam = cam_pos - p;
  float weight = NdotV * NdotL / dot(delta_cam, delta_cam) * occlusion * shading * photo_gate;

  uchar3 photo_bytes = convert_uchar3_sat(floor(sample * 256.0f));
  uint w0 = (uint)photo_bytes.x | ((uint)photo_bytes.y << 8) | ((uint)photo_bytes.z << 16) |
            ((camera_index & 0xFFu) << 24);
  uint w1 = camera_index >> 8;
  // The stored witness is light-independent: occlusion and photo_gate only. The
  // weight depends on the light, so it is recomputed in EXPAND_KERNEL from the
  // candidate light; only its sum over all cameras is kept here for coverage.
  uint w2 = floatToHalf(occlusion) | (floatToHalf(photo_gate) << 16);

  // One work item per texel per camera, and cameras are dispatched sequentially,
  // so plain read/modify/write is race-free. When the list is full, replace the
  // weakest witness by its light-independent rank instead of dropping it.
  float rank = occlusion * photo_gate;
  uint count = counts[i];
  bool store = true;
  uint slot = count;
  if (count < cap) {
    counts[i] = count + 1u;
  } else {
    slot = 0u;
    uint packed = pool[(i * cap + 0u) * 3u + 2u];
    float weakest = halfToFloat(packed & 0xFFFFu) * halfToFloat(packed >> 16u);
    for (uint k = 1u; k < cap; ++k) {
      packed = pool[(i * cap + k) * 3u + 2u];
      float candidate = halfToFloat(packed & 0xFFFFu) * halfToFloat(packed >> 16u);
      if (candidate < weakest) {
        weakest = candidate;
        slot = k;
      }
    }
    store = rank > weakest;
  }
  if (store) {
    __global uint *dst = pool + ((size_t)i * cap + slot) * 3u;
    dst[0] = w0; dst[1] = w1; dst[2] = w2;
  }
  weights[i] += weight;
}
"""
)


EXPAND_KERNEL = (
    _COMMON_HELPERS
    + b"""
// Recomputes the light-dependent witness (directions, radiance, weight) for the
// candidate light and packs it into the 7-word expanded layout. The compact pool
// only carries light-independent data (photo, image index, occlusion,
// photo_gate), so a candidate light only needs this cheap pass, not a gather.
__kernel void expand(
    __global const uint *pool,
    __global const uint *counts,
    __global const float *positions,
    __global const float *normals,
    __global const float *lights,
    __global uint *expanded,
    __global float *weights,
    const uint tile_offset,
    const uint tile_count,
    const uint cap,
    const float shading_gate)
{
  uint i = get_global_id(0);
  if (i >= tile_count) return;
  uint gidx = tile_offset + i;
  uint c = counts[gidx];
  if (c == 0u) {
    weights[gidx] = 0.0f;
    return;
  }
  float3 P = (float3)(positions[gidx * 3u], positions[gidx * 3u + 1u], positions[gidx * 3u + 2u]);
  float3 N = (float3)(normals[gidx * 3u], normals[gidx * 3u + 1u], normals[gidx * 3u + 2u]);
  uint source = gidx * cap * 3u;
  uint dest = i * cap * 7u;
  float weight_sum = 0.0f;
  for (uint k = 0u; k < c; ++k) {
    uint off = source + k * 3u;
    uint w0 = pool[off];
    uint image = ((w0 >> 24u) & 0xFFu) | (pool[off + 1u] << 8u);
    uint packed = pool[off + 2u];
    float occlusion = halfToFloat(packed & 0xFFFFu);
    float photo_gate = halfToFloat(packed >> 16u);
    float3 cam = (float3)(lights[image * 7u], lights[image * 7u + 1u], lights[image * 7u + 2u]);
    float3 lp = (float3)(lights[image * 7u + 3u], lights[image * 7u + 4u], lights[image * 7u + 5u]);
    float mult = lights[image * 7u + 6u];
    float3 V = normalize(cam - P);
    float3 L = normalize(lp - P);
    float3 dl = lp - P;
    float3 R = (float3)(mult) / dot(dl, dl);
    float NdotL = fmax(0.0f, dot(N, L));
    float NdotV = fmax(0.0f, dot(N, V));
    float shading = smoothstep01(NdotL / shading_gate) * smoothstep01(NdotV / shading_gate);
    float3 delta_cam = cam - P;
    float weight = (shading * photo_gate < 1e-4f)
                       ? 0.0f
                       : NdotV * NdotL / dot(delta_cam, delta_cam) * occlusion * shading * photo_gate;
    uint d = dest + k * 7u;
    expanded[d] = w0 & 0xFFFFFFu;
    expanded[d + 1u] = floatToHalf(R.x) | (floatToHalf(R.y) << 16);
    expanded[d + 2u] = floatToHalf(R.z) | (floatToHalf(V.x) << 16);
    expanded[d + 3u] = floatToHalf(V.y) | (floatToHalf(V.z) << 16);
    expanded[d + 4u] = floatToHalf(L.x) | (floatToHalf(L.y) << 16);
    expanded[d + 5u] = floatToHalf(L.z);
    expanded[d + 6u] = as_uint(weight);
    weight_sum += weight;
  }
  weights[gidx] = weight_sum;
}
"""
)


BLUR_KERNEL = (
    _COMMON_HELPERS
    + b"""
__kernel void blur(
    __global const float *pos,
    __global const uint *cell_start,
    __global const uint *cell_items,
    __global const float *geom_normals,
    __global const float *geom_tangents,
    __global const float *map_a,
    __global const float *map_r,
    __global const float *map_m,
    __global const float *map_n,
    __global float *dst_a,
    __global float *dst_r,
    __global float *dst_m,
    __global float *dst_n,
    const uint pixel_count,
    const int gx, const int gy, const int gz,
    const float ox, const float oy, const float oz,
    const float cell,
    const float sigma_sq2,
    const float search_radius_sq)
{
  uint i = get_global_id(0);
  if (i >= pixel_count) return;

  float3 p = (float3)(pos[i * 3u], pos[i * 3u + 1u], pos[i * 3u + 2u]);
  if (isnan(p.x) || isnan(p.y) || isnan(p.z)) {
    for (int k = 0; k < 4; ++k) { dst_a[i * 4u + k] = map_a[i * 4u + k]; dst_n[i * 4u + k] = map_n[i * 4u + k]; }
    dst_r[i] = map_r[i];
    dst_m[i] = map_m[i];
    return;
  }

  float3 geom_normal = normalize((float3)(geom_normals[i * 3u], geom_normals[i * 3u + 1u], geom_normals[i * 3u + 2u]));
  float3 tangent = normalize((float3)(geom_tangents[i * 3u], geom_tangents[i * 3u + 1u], geom_tangents[i * 3u + 2u]));
  if (length(tangent) < 1e-4f) {
    tangent = cross(geom_normal, (float3)(0.0f, 0.0f, 1.0f));
    if (length(tangent) < 1e-4f) tangent = (float3)(1.0f, 0.0f, 0.0f);
  }
  tangent = normalize(tangent);
  float3 bitangent = normalize(cross(geom_normal, tangent));

  int cx = (int)floor((p.x - ox) / cell);
  int cy = (int)floor((p.y - oy) / cell);
  int cz = (int)floor((p.z - oz) / cell);
  float4 accA = (float4)(0.0f);
  float accR = 0.0f, accM = 0.0f, accW = 0.0f, accWN = 0.0f;
  float3 accN = (float3)(0.0f);

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int ccx = cx + dx, ccy = cy + dy, ccz = cz + dz;
        if (ccx < 0 || ccy < 0 || ccz < 0 || ccx >= gx || ccy >= gy || ccz >= gz) continue;
        uint id = (uint)(ccx + ccy * gx + ccz * gx * gy);
        for (uint k = cell_start[id]; k < cell_start[id + 1u]; ++k) {
          uint j = cell_items[k];
          float3 d = (float3)(pos[j * 3u], pos[j * 3u + 1u], pos[j * 3u + 2u]) - p;
          float dsq = dot(d, d);
          if (dsq > search_radius_sq) continue;
          float w = exp(-dsq / sigma_sq2);
          accA += (float4)(map_a[j * 4u], map_a[j * 4u + 1u], map_a[j * 4u + 2u], map_a[j * 4u + 3u]) * w;
          accR += map_r[j] * w;
          accM += map_m[j] * w;
          accW += w;
          float3 jn = (float3)(geom_normals[j * 3u], geom_normals[j * 3u + 1u], geom_normals[j * 3u + 2u]);
          if (!isnan(jn.x) && !isnan(jn.y) && !isnan(jn.z)) {
            float3 jt = normalize((float3)(geom_tangents[j * 3u], geom_tangents[j * 3u + 1u], geom_tangents[j * 3u + 2u]));
            if (length(jt) < 1e-4f) {
              jt = cross(jn, (float3)(0.0f, 0.0f, 1.0f));
              if (length(jt) < 1e-4f) jt = (float3)(1.0f, 0.0f, 0.0f);
            }
            float3 sample = (float3)(map_n[j * 4u], map_n[j * 4u + 1u], map_n[j * 4u + 2u]);
            accN += getNormal(normalize(jn), sample, normalize(jt)) * w;
            accWN += w;
          }
        }
      }
    }
  }

  float invW = 1.0f / fmax(accW, 1e-6f);
  dst_a[i * 4u] = accA.x * invW;
  dst_a[i * 4u + 1u] = accA.y * invW;
  dst_a[i * 4u + 2u] = accA.z * invW;
  dst_a[i * 4u + 3u] = accA.w * invW;
  dst_r[i] = accR * invW;
  dst_m[i] = accM * invW;
  if (accWN > 0.0f) {
    float3 n_world = accN / accWN;
    float3 n_tangent = (float3)(dot(tangent, n_world), dot(bitangent, n_world), dot(geom_normal, n_world));
    float3 encoded = normalize(n_tangent) * 0.5f + 0.5f;
    dst_n[i * 4u] = encoded.x;
    dst_n[i * 4u + 1u] = encoded.y;
    dst_n[i * 4u + 2u] = encoded.z;
    dst_n[i * 4u + 3u] = 1.0f;
  } else {
    for (int k = 0; k < 4; ++k) dst_n[i * 4u + k] = map_n[i * 4u + k];
  }
}
"""
)


OPTIMIZER_KERNEL = (
    _COMMON_HELPERS
    + b"""
#define PBR_PI 3.14159265358979323846f
#define PBR_DIELECTRIC_F0 0.04f
#define PBR_MIN_ROUGHNESS 0.04f
#define PBR_MAX_ROUGHNESS 1.0f

// Expanded witness layout (7 words), identical to the base program's Witness.
#define WIT_PHOTO(e, b, i) ((float3)((float)(e[(b) + (i) * 7u] & 255u), (float)((e[(b) + (i) * 7u] >> 8u) & 255u), (float)((e[(b) + (i) * 7u] >> 16u) & 255u)) / 255.0f)
#define WIT_WEIGHT(e, b, i) (as_float(e[(b) + (i) * 7u + 6u]))
#define WIT_RADIANCE(e, b, i) ((float3)(halfToFloat(e[(b) + (i) * 7u + 1u]), halfToFloat(e[(b) + (i) * 7u + 1u] >> 16u), halfToFloat(e[(b) + (i) * 7u + 2u])))
#define WIT_VIEW(e, b, i) ((float3)(halfToFloat(e[(b) + (i) * 7u + 2u] >> 16u), halfToFloat(e[(b) + (i) * 7u + 3u]), halfToFloat(e[(b) + (i) * 7u + 3u] >> 16u)))
#define WIT_LIGHT(e, b, i) ((float3)(halfToFloat(e[(b) + (i) * 7u + 4u]), halfToFloat(e[(b) + (i) * 7u + 4u] >> 16u), halfToFloat(e[(b) + (i) * 7u + 5u])))

typedef struct { float3 albedo; float roughness; float metallic; float3 normal; } Sample;
typedef struct { int max_iters; float step_alpha, min_step, grad_tol, loss_tol, rms_decay, rms_eps; int use_long_bb; } OptParams;

static inline Sample normalizeSample(Sample s) {
  s.albedo = clamp(s.albedo, 0.0f, 1.0f);
  s.roughness = clamp(s.roughness, 0.0f, 1.0f);
  s.metallic = clamp(s.metallic, 0.0f, 1.0f);
  s.normal = (normalize(s.normal * 2.0f - (float3)(1.0f)) + (float3)(1.0f)) * 0.5f;
  return s;
}
static inline float3 computeF0(float3 albedo, float metallic) {
  return (float3)(PBR_DIELECTRIC_F0) * (1.0f - metallic) + albedo * metallic;
}
static inline float3 fresnelSchlick(float cosTheta, float3 F0) {
  return F0 + ((float3)(1.0f) - F0) * pow(1.0f - cosTheta, 5.0f);
}
static inline float distributionGGX(float NdotH, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH2 = NdotH * NdotH;
  float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
  return a2 / (PBR_PI * denom * denom);
}
static inline float G1(float NdotX, float a) {
  if (NdotX <= 0.0f) return 0.0f;
  float k = (a + 1.0f) * (a + 1.0f) / 8.0f;
  return NdotX / (NdotX * (1.0f - k) + k);
}
static inline float geometrySmith(float NdotV, float NdotL, float roughness) {
  float a = roughness * roughness;
  return G1(NdotV, a) * G1(NdotL, a);
}
static inline float3 toneMap(float3 color) { return color / (color + (float3)(1.0f)); }
static inline float3 inverseToneMap(float3 color) {
  color = clamp(color, (float3)(0.0f), (float3)(0.999f));
  return color / ((float3)(1.0f) - color);
}
static inline float3 renderColor(float3 albedo, float roughness, float metallic, float3 normal,
                          float ambient, float3 view_dir, float3 light_dir, float3 radiance) {
  float r = clamp(roughness, PBR_MIN_ROUGHNESS, PBR_MAX_ROUGHNESS);
  float m = clamp(metallic, 0.0f, 1.0f);
  float NdotL = max(0.0f, dot(normal, light_dir));
  float NdotV = max(0.0f, dot(normal, view_dir));
  float3 diffuse = (float3)(0.0f);
  float3 specular = (float3)(0.0f);
  if (NdotL > 1e-4f && NdotV > 1e-4f) {
    float3 H = normalize(view_dir + light_dir);
    float NdotH = max(0.0f, dot(normal, H));
    float VdotH = max(0.0f, dot(view_dir, H));
    float3 F0 = computeF0(albedo, m);
    float3 F = fresnelSchlick(VdotH, F0);
    float D = distributionGGX(NdotH, r);
    float G = geometrySmith(NdotV, NdotL, r);
    specular = (D * G * F) / (4.0f * NdotV + 1e-5f);
    float3 kD = ((float3)(1.0f) - F) * (1.0f - m);
    diffuse = albedo * kD / PBR_PI;
  }
  float3 direct = (diffuse + specular) * NdotL * radiance;
  float3 ambient_term = albedo * ambient * (1.0f - metallic);
  return clamp(toneMap(ambient_term + direct), (float3)(0.0f), (float3)(1.0f));
}

static inline float3 photoDifference(int count, Sample s, float3 normal_world, float ambient,
                              __global const uint *exp, uint base) {
  float3 avg_error = (float3)(0.0f);
  float weight_sum = 0.0f;
  for (int i = 0; i < count; ++i) {
    float3 reconstructed = renderColor(s.albedo, s.roughness, s.metallic, normal_world, ambient,
                                       WIT_VIEW(exp, base, i), WIT_LIGHT(exp, base, i), WIT_RADIANCE(exp, base, i));
    avg_error += fabs(reconstructed - WIT_PHOTO(exp, base, i)) * WIT_WEIGHT(exp, base, i);
    weight_sum += WIT_WEIGHT(exp, base, i);
  }
  return avg_error / weight_sum;
}
static inline float sampleLoss(int count, Sample s, float3 normal_world, float ambient,
                        __global const uint *exp, uint base) {
  float3 e = photoDifference(count, s, normal_world, ambient, exp, base);
  return (e.x + e.y + e.z) / 3.0f;
}

typedef struct {
  float r_eff, m_raw, m_eff, dm_eff_dm;
  float r2, a2, k, da2_dr, dk_dr;
  float3 F0, u, N;
  float s_n, t_n;
} BRDFParams;
static inline BRDFParams computeBRDFParams(Sample s, float3 geom_normal, float3 tangent) {
  BRDFParams p;
  p.r_eff = clamp(s.roughness, PBR_MIN_ROUGHNESS, PBR_MAX_ROUGHNESS);
  p.m_raw = s.metallic;
  p.m_eff = clamp(p.m_raw, 0.0f, 1.0f);
  p.dm_eff_dm = (p.m_raw >= 0.0f && p.m_raw <= 1.0f) ? 1.0f : 0.0f;
  p.r2 = p.r_eff * p.r_eff;
  p.a2 = p.r2 * p.r2;
  p.k = (p.r2 + 1.0f) * (p.r2 + 1.0f) / 8.0f;
  p.da2_dr = 4.0f * p.r2 * p.r_eff;
  p.dk_dr = (p.r2 + 1.0f) * p.r_eff / 2.0f;
  p.F0 = (float3)(PBR_DIELECTRIC_F0) * (1.0f - p.m_eff) + s.albedo * p.m_eff;
  float3 v_tangent = s.normal * 2.0f - (float3)(1.0f);
  p.s_n = max(length(v_tangent), 1e-10f);
  p.u = v_tangent / p.s_n;
  float3 bitangent = normalize(cross(geom_normal, tangent));
  float3 y = tangent * p.u.x + bitangent * p.u.y + geom_normal * p.u.z;
  p.t_n = max(length(y), 1e-10f);
  p.N = y / p.t_n;
  return p;
}

static inline Sample sampleGradientAnalytical(int count, Sample s, float ambient, float3 geom_normal,
                                       float3 tangent, __global const uint *exp, uint base) {
  float weight_sum = 0.0f;
  for (int i = 0; i < count; ++i) weight_sum += WIT_WEIGHT(exp, base, i);
  Sample grad;
  grad.albedo = (float3)(0.0f); grad.roughness = 0.0f; grad.metallic = 0.0f; grad.normal = (float3)(0.0f);
  if (weight_sum < 1e-6f) return grad;

  BRDFParams p = computeBRDFParams(s, geom_normal, tangent);
  float3 N = p.N;
  float3 grad_a = (float3)(0.0f);
  float grad_r = 0.0f, grad_m = 0.0f;
  float3 grad_n = (float3)(0.0f);

  for (int i = 0; i < count; ++i) {
    float3 V = WIT_VIEW(exp, base, i);
    float3 L = WIT_LIGHT(exp, base, i);
    float3 R = WIT_RADIANCE(exp, base, i);
    float3 photo = WIT_PHOTO(exp, base, i);
    float weight = WIT_WEIGHT(exp, base, i);

    float3 H = normalize(V + L);
    float NdotL_val = max(0.0f, dot(N, L));
    float NdotV_val = max(0.0f, dot(N, V));
    float NdotH_val = max(0.0f, dot(N, H));
    float VdotH = max(0.0f, dot(V, H));

    float3 S, dS_dA, dS_dm, dS_dr, F, diffuse, spec, dspec_dN_base;
    if (NdotL_val <= 1e-4f || NdotV_val <= 1e-4f) {
      S = s.albedo * ambient * (1.0f - p.m_raw);
      dS_dA = (float3)(ambient * (1.0f - p.m_raw));
      dS_dm = -s.albedo * ambient;
      dS_dr = (float3)(0.0f);
      F = diffuse = spec = (float3)(0.0f);
      dspec_dN_base = (float3)(0.0f);
    } else {
      float f5 = pow(1.0f - VdotH, 5.0f);
      F = p.F0 + ((float3)(1.0f) - p.F0) * f5;
      float3 kD = ((float3)(1.0f) - F) * (1.0f - p.m_eff);
      diffuse = s.albedo * kD / PBR_PI;
      float T_d = NdotH_val * NdotH_val * (p.a2 - 1.0f) + 1.0f;
      float D = p.a2 / (PBR_PI * T_d * T_d);
      float denom_g1_V = NdotV_val * (1.0f - p.k) + p.k;
      float denom_g1_L = NdotL_val * (1.0f - p.k) + p.k;
      float g1_V = NdotV_val / denom_g1_V;
      float g1_L = NdotL_val / denom_g1_L;
      float G = g1_V * g1_L;
      float denom_spec = 4.0f * NdotV_val + 1e-5f;
      spec = (D * G / denom_spec) * F;
      float3 common_term = NdotL_val * R;
      S = s.albedo * ambient * (1.0f - p.m_raw) + (diffuse + spec) * common_term;
      float NdotH2 = NdotH_val * NdotH_val;
      float dD_da2 = (1.0f - NdotH2 * (1.0f + p.a2)) / (PBR_PI * T_d * T_d * T_d);
      float dD_dr = dD_da2 * p.da2_dr;
      float dD_dNdotH = -4.0f * p.a2 * NdotH_val * (p.a2 - 1.0f) / (PBR_PI * T_d * T_d * T_d);
      float3 dD_dN = H * dD_dNdotH;
      float dg1_V_dk = -NdotV_val * (1.0f - NdotV_val) / (denom_g1_V * denom_g1_V);
      float dg1_L_dk = -NdotL_val * (1.0f - NdotL_val) / (denom_g1_L * denom_g1_L);
      float dG_dr = p.dk_dr * (g1_L * dg1_V_dk + g1_V * dg1_L_dk);
      float dg1_V_dx = p.k / (denom_g1_V * denom_g1_V);
      float dg1_L_dx = p.k / (denom_g1_L * denom_g1_L);
      float3 dG_dN = g1_L * dg1_V_dx * V + g1_V * dg1_L_dx * L;
      float d_DG_dr = G * dD_dr + D * dG_dr;
      float3 d_DG_dN = G * dD_dN + D * dG_dN;
      float dF_dA = p.m_eff * (1.0f - f5);
      float3 dF_dm = (s.albedo - (float3)(PBR_DIELECTRIC_F0)) * (1.0f - f5);
      float DG_over_denom = D * G / denom_spec;
      float3 dspec_dA = (float3)(DG_over_denom * dF_dA);
      float3 dspec_dm = DG_over_denom * dF_dm;
      float3 dspec_dr = (d_DG_dr / denom_spec) * F;
      dspec_dN_base = d_DG_dN / denom_spec - (D * G) * (4.0f * V) / (denom_spec * denom_spec);
      float3 dkD_dm = -((float3)(1.0f) - F) - (1.0f - p.m_eff) * dF_dm;
      float3 ddiffuse_dA = (kD + s.albedo * (-(1.0f - p.m_eff)) * dF_dA) / PBR_PI;
      float3 ddiffuse_dm = s.albedo * dkD_dm / PBR_PI;
      dS_dA = (float3)(ambient * (1.0f - p.m_raw)) + (ddiffuse_dA + dspec_dA) * common_term;
      dS_dm = -s.albedo * ambient + (ddiffuse_dm + dspec_dm) * common_term * p.dm_eff_dm;
      dS_dr = dspec_dr * common_term;
    }

    float3 r = S / (S + (float3)(1.0f));
    float3 r_clamped = clamp(r, (float3)(0.0f), (float3)(1.0f));
    float3 dL_dr = sign(r_clamped - photo) * (weight / (3.0f * weight_sum));
    float3 dL_dS;
    dL_dS.x = (r_clamped.x > 0.0f && r_clamped.x < 1.0f) ? dL_dr.x / ((S.x + 1.0f) * (S.x + 1.0f)) : 0.0f;
    dL_dS.y = (r_clamped.y > 0.0f && r_clamped.y < 1.0f) ? dL_dr.y / ((S.y + 1.0f) * (S.y + 1.0f)) : 0.0f;
    dL_dS.z = (r_clamped.z > 0.0f && r_clamped.z < 1.0f) ? dL_dr.z / ((S.z + 1.0f) * (S.z + 1.0f)) : 0.0f;

    grad_a += dL_dS * dS_dA;
    grad_r += dot(dL_dS, dS_dr);
    grad_m += dot(dL_dS, dS_dm);
    float3 wgt = dL_dS * R;
    grad_n += NdotL_val * dot(wgt, F) * dspec_dN_base + dot(wgt, diffuse + spec) * L;
  }

  if (s.roughness < PBR_MIN_ROUGHNESS || s.roughness > PBR_MAX_ROUGHNESS) grad_r = 0.0f;
  grad.albedo = grad_a;
  grad.roughness = grad_r;
  grad.metallic = grad_m;
  float3 t1 = grad_n - N * dot(N, grad_n);
  float3 bitangent = normalize(cross(geom_normal, tangent));
  float3 t2 = (float3)(dot(tangent, t1), dot(bitangent, t1), dot(geom_normal, t1));
  float3 t3 = t2 - p.u * dot(p.u, t2);
  grad.normal = (2.0f / (p.s_n * p.t_n)) * t3;
  return grad;
}

static inline float3 albedoEstimate(int count, float3 geom_normal, float ambient,
                             __global const uint *exp, uint base) {
  float3 albedo_sum = (float3)(0.0f);
  float weight_sum = 0.0f;
  for (int i = 0; i < count; ++i) {
    float3 V = WIT_VIEW(exp, base, i);
    float3 L = WIT_LIGHT(exp, base, i);
    float3 R = WIT_RADIANCE(exp, base, i);
    float weight = WIT_WEIGHT(exp, base, i);
    float NdotL = max(0.0f, dot(geom_normal, L));
    float NdotV = max(0.0f, dot(geom_normal, V));
    if (NdotL < 0.1f || NdotV < 0.1f) continue;
    float3 H = normalize(V + L);
    float NdotH = max(0.0f, dot(geom_normal, H));
    float VdotH = max(0.0f, dot(V, H));
    float3 F0 = (float3)(PBR_DIELECTRIC_F0);
    float3 F = fresnelSchlick(VdotH, F0);
    float3 kD = ((float3)(1.0f) - F);
    float3 diffuse = kD / PBR_PI;
    float D = distributionGGX(NdotH, 0.3f);
    float G = geometrySmith(NdotV, NdotL, 0.3f);
    float3 specular = ((D * G * F) / (4.0f * NdotV + 1e-5f) * NdotL) * R;
    float3 shaded = inverseToneMap(WIT_PHOTO(exp, base, i));
    if (max(specular.x, max(specular.y, specular.z)) > 0.3f * max(shaded.x, max(shaded.y, shaded.z))) continue;
    float3 denom = (float3)(ambient) + (diffuse * NdotL) * R;
    if (min(denom.x, min(denom.y, denom.z)) < 1e-4f) continue;
    float3 est = (shaded - specular) / denom;
    if (any(isnan(est))) continue;
    est = clamp(est, (float3)(-0.1f), (float3)(1.1f));
    albedo_sum += est * weight;
    weight_sum += weight;
  }
  if (weight_sum < 1e-6f) return (float3)(0.75f);
  return albedo_sum / weight_sum;
}

static inline Sample optimizePBR(int count, float3 geom_normal, float3 tangent, float ambient, Sample init,
                          OptParams p, __global const uint *exp, uint base) {
  Sample x = normalizeSample(init);
  float3 normal_world = getNormal(geom_normal, x.normal, tangent);
  float loss = sampleLoss(count, x, normal_world, ambient, exp, base);
  float3 rms_a = (float3)(0.0f);
  float rms_r = 0.0f, rms_m = 0.0f;
  float3 rms_n = (float3)(0.0f);
  Sample x_prev = x;
  Sample g_prev = sampleGradientAnalytical(count, x, ambient, geom_normal, tangent, exp, base);
  float alpha = p.step_alpha;
  for (int iter = 0; iter < p.max_iters; ++iter) {
    Sample g = sampleGradientAnalytical(count, x, ambient, geom_normal, tangent, exp, base);
    float max_g = max(max(max(fabs(g.albedo.x), fabs(g.albedo.y)), max(fabs(g.albedo.z), fabs(g.roughness))),
                      max(max(fabs(g.metallic), fabs(g.normal.x)), max(fabs(g.normal.y), fabs(g.normal.z))));
    if (max_g * alpha < p.grad_tol) break;
    rms_a = p.rms_decay * rms_a + (1.0f - p.rms_decay) * g.albedo * g.albedo;
    rms_r = p.rms_decay * rms_r + (1.0f - p.rms_decay) * g.roughness * g.roughness;
    rms_m = p.rms_decay * rms_m + (1.0f - p.rms_decay) * g.metallic * g.metallic;
    rms_n = p.rms_decay * rms_n + (1.0f - p.rms_decay) * g.normal * g.normal;
    Sample sg;
    sg.albedo = g.albedo / sqrt(rms_a + p.rms_eps);
    sg.roughness = g.roughness / sqrt(rms_r + p.rms_eps);
    sg.metallic = g.metallic / sqrt(rms_m + p.rms_eps);
    sg.normal = g.normal / sqrt(rms_n + p.rms_eps);
    if (iter > 0) {
      float3 dx_a = x.albedo - x_prev.albedo;
      float dx_r = x.roughness - x_prev.roughness;
      float dx_m = x.metallic - x_prev.metallic;
      float3 dx_n = x.normal - x_prev.normal;
      float3 dg_a = g.albedo - g_prev.albedo;
      float dg_r = g.roughness - g_prev.roughness;
      float dg_m = g.metallic - g_prev.metallic;
      float3 dg_n = g.normal - g_prev.normal;
      float dx_dg = dot(dx_a, dg_a) + dx_r * dg_r + dx_m * dg_m + dot(dx_n, dg_n);
      float dg2 = dot(dg_a, dg_a) + dg_r * dg_r + dg_m * dg_m + dot(dg_n, dg_n);
      if (fabs(dx_dg) > 1e-12f && dg2 > 1e-12f) {
        if (p.use_long_bb == 1) alpha = fabs(dx_dg) / dg2;
        else {
          float dx2 = dot(dx_a, dx_a) + dx_r * dx_r + dx_m * dx_m + dot(dx_n, dx_n);
          alpha = dx2 / fabs(dx_dg);
        }
      }
    }
    Sample x_new;
    x_new.albedo = x.albedo - alpha * sg.albedo;
    x_new.roughness = x.roughness - alpha * sg.roughness;
    x_new.metallic = x.metallic - alpha * sg.metallic;
    x_new.normal = x.normal - alpha * sg.normal;
    x_new = normalizeSample(x_new);
    float3 new_normal_world = getNormal(geom_normal, x_new.normal, tangent);
    float loss_new = sampleLoss(count, x_new, new_normal_world, ambient, exp, base);
    int backtracks = 0;
    const int max_back = 10;
    while (loss_new >= loss && backtracks < max_back && alpha > p.min_step) {
      alpha *= 0.5f;
      x_new.albedo = x.albedo - alpha * sg.albedo;
      x_new.roughness = x.roughness - alpha * sg.roughness;
      x_new.metallic = x.metallic - alpha * sg.metallic;
      x_new.normal = x.normal - alpha * sg.normal;
      x_new = normalizeSample(x_new);
      new_normal_world = getNormal(geom_normal, x_new.normal, tangent);
      loss_new = sampleLoss(count, x_new, new_normal_world, ambient, exp, base);
      ++backtracks;
    }
    if (backtracks == max_back || alpha <= p.min_step) break;
    float improvement = (loss - loss_new) / loss;
    if (improvement < p.loss_tol && max_g * alpha < p.grad_tol * 10.0f) break;
    x_prev = x;
    g_prev = g;
    x = x_new;
    loss = loss_new;
  }
  return x;
}

static inline float nextNoise(uint *state) {
  *state ^= *state << 13u;
  *state ^= *state >> 17u;
  *state ^= *state << 5u;
  return (float)(*state & 0xFFFFFFu) / 8388608.0f - 1.0f;
}
static inline Sample jitteredStart(Sample base, float jitter, uint *state) {
  Sample j = base;
  j.albedo += (float3)(nextNoise(state), nextNoise(state), nextNoise(state)) * jitter;
  j.metallic += nextNoise(state) * jitter;
  j.roughness += nextNoise(state) * jitter;
  j.normal += (float3)(nextNoise(state), nextNoise(state), nextNoise(state)) * jitter;
  return j;
}

__kernel void optimize(
    __global const float *positions,
    __global const float *normals,
    __global const float *tangents,
    __global const float *lights,
    __global const uint *exp,
    __global const uint *counts,
    __global float *albedo,
    __global float *roughness,
    __global float *metallic,
    __global float *normal,
    __global float *errors,
    __global const float *candidates,
    __global const float *params,
    const uint pixel_count,
    const uint cap,
    const uint tile_offset,
    const int estimate_albedo,
    const int use_candidates,
    const uint subpass,
    const uint candidate_count,
    const int max_iters)
{
  uint idx = get_global_id(0);
  if (idx >= pixel_count) return;
  uint gidx = tile_offset + idx;
  uint c = counts[gidx];
  if (c == 0u) return;
  uint base = idx * cap * 7u;
  float3 geom_normal = (float3)(normals[gidx * 3u], normals[gidx * 3u + 1u], normals[gidx * 3u + 2u]);
  float3 tangent = (float3)(tangents[gidx * 3u], tangents[gidx * 3u + 1u], tangents[gidx * 3u + 2u]);
  float ambient = params[0];
  float start_jitter = params[1];
  OptParams op;
  op.max_iters = max_iters;
  op.step_alpha = params[2];
  op.min_step = params[3];
  op.grad_tol = params[4];
  op.loss_tol = params[5];
  op.rms_decay = params[6];
  op.rms_eps = params[7];
  op.use_long_bb = (int)params[8];

  Sample init;
  init.albedo = (estimate_albedo == 1)
                    ? albedoEstimate((int)c, geom_normal, ambient, exp, base)
                    : (float3)(albedo[gidx * 4u], albedo[gidx * 4u + 1u], albedo[gidx * 4u + 2u]);
  init.roughness = roughness[gidx];
  init.metallic = metallic[gidx];
  init.normal = (float3)(normal[gidx * 4u], normal[gidx * 4u + 1u], normal[gidx * 4u + 2u]);

  Sample best = init;
  if (use_candidates == 1) {
    float3 bn = getNormal(geom_normal, best.normal, tangent);
    float best_loss = sampleLoss((int)c, best, bn, ambient, exp, base);
    for (uint ci = 0; ci < candidate_count; ++ci) {
      Sample cand = best;
      cand.roughness = candidates[ci];
      uint noise = gidx * 2654435761u + subpass * 2246822519u + 1u;
      cand = jitteredStart(cand, start_jitter, &noise);
      cand = optimizePBR((int)c, geom_normal, tangent, ambient, cand, op, exp, base);
      float3 cn = getNormal(geom_normal, cand.normal, tangent);
      float cl = sampleLoss((int)c, cand, cn, ambient, exp, base);
      if (cl < best_loss) { best_loss = cl; best = cand; }
    }
  } else {
    best = optimizePBR((int)c, geom_normal, tangent, ambient, best, op, exp, base);
  }

  albedo[gidx * 4u] = best.albedo.x;
  albedo[gidx * 4u + 1u] = best.albedo.y;
  albedo[gidx * 4u + 2u] = best.albedo.z;
  albedo[gidx * 4u + 3u] = 1.0f;
  roughness[gidx] = best.roughness;
  metallic[gidx] = best.metallic;
  normal[gidx * 4u] = best.normal.x;
  normal[gidx * 4u + 1u] = best.normal.y;
  normal[gidx * 4u + 2u] = best.normal.z;
  normal[gidx * 4u + 3u] = 1.0f;

  float3 nw = getNormal(geom_normal, best.normal, tangent);
  float3 e = photoDifference((int)c, best, nw, ambient, exp, base);
  errors[gidx * 4u] = e.x;
  errors[gidx * 4u + 1u] = e.y;
  errors[gidx * 4u + 2u] = e.z;
  errors[gidx * 4u + 3u] = 0.0f;
}
"""
)


class WitnessGatherer(_OpenCLProgram):
    """Runs the gather kernel once per camera into a shared per-texel pool."""

    def __init__(self, device, texel_count, cap):
        super().__init__(device)
        self.texel_count = texel_count
        self.cap = cap
        self._program, kernel = _build_program(
            self._cl, self._context, device, GATHER_KERNEL, "gather"
        )
        self._gather = _Kernel(self._cl, self._queue, kernel)
        self._positions = self._buffer(texel_count * 3 * 4, CL_MEM_READ_ONLY)
        self._normals = self._buffer(texel_count * 3 * 4, CL_MEM_READ_ONLY)
        self._counts = self._buffer(
            texel_count * 4,
            CL_MEM_READ_WRITE,
            np.zeros(texel_count, np.uint32).tobytes(),
        )
        self._weights = self._buffer(
            texel_count * 4,
            CL_MEM_READ_WRITE,
            np.zeros(texel_count, np.float32).tobytes(),
        )
        self._pool = self._buffer(
            texel_count * cap * WITNESS_WORDS * 4, CL_MEM_READ_WRITE
        )

    def set_geometry(self, positions, normals):
        self._write(self._positions, positions)
        self._write(self._normals, normals)
        self._cl.clFinish(self._queue)

    def run_camera(
        self,
        lut,
        lut_grid,
        depth,
        cam_w,
        cam_h,
        photo,
        params,
        camera_index,
        occlusion_radius,
    ):
        lut_memory = self._buffer(lut.nbytes, CL_MEM_READ_ONLY, lut.tobytes())
        depth_memory = self._buffer(depth.nbytes, CL_MEM_READ_ONLY, depth.tobytes())
        photo_memory = self._buffer(photo.nbytes, CL_MEM_READ_ONLY, photo.tobytes())
        params_memory = self._buffer(params.nbytes, CL_MEM_READ_ONLY, params.tobytes())
        try:
            kernel = self._gather
            kernel.set_buffer(0, self._positions)
            kernel.set_buffer(1, self._normals)
            kernel.set_buffer(2, lut_memory)
            kernel.set_buffer(3, depth_memory)
            kernel.set_buffer(4, photo_memory)
            kernel.set_buffer(5, self._weights)
            kernel.set_buffer(6, self._pool)
            kernel.set_buffer(7, self._counts)
            kernel.set_buffer(8, params_memory)
            kernel.set_uint(9, self.texel_count)
            kernel.set_uint(10, self.cap)
            kernel.set_uint(11, lut_grid)
            kernel.set_uint(12, cam_w)
            kernel.set_uint(13, cam_h)
            kernel.set_uint(14, photo.shape[1])
            kernel.set_uint(15, photo.shape[0])
            kernel.set_int(16, occlusion_radius)
            kernel.set_uint(17, camera_index)
            kernel.launch(((self.texel_count + 63) // 64) * 64)
        finally:
            for memory in (lut_memory, depth_memory, photo_memory, params_memory):
                self._cl.clReleaseMemObject(memory)

    def coverage(self):
        weights = self._read(self._weights, self.texel_count * 4, np.float32)
        maximum = float(weights.max()) if weights.size else 0.0
        return weights / maximum if maximum > 0.0 else weights

    def read_pool(self):
        size = self.texel_count * self.cap * WITNESS_WORDS
        pool = self._read(self._pool, size * 4, np.uint32)
        return pool.reshape(self.texel_count, self.cap, WITNESS_WORDS)

    def read_counts(self):
        return self._read(self._counts, self.texel_count * 4, np.uint32)

    def destroy(self):
        self._cl.clFinish(self._queue)
        for memory in (
            self._positions,
            self._normals,
            self._counts,
            self._weights,
            self._pool,
        ):
            self._cl.clReleaseMemObject(memory)
        self._cl.clReleaseKernel(self._gather.handle)
        self._cl.clReleaseProgram(self._program)


def _world_texel_size(positions):
    valid = ~np.isnan(positions[..., 0])
    edges = []
    horizontal = valid[:, :-1] & valid[:, 1:]
    if horizontal.any():
        edges.append(
            np.linalg.norm(positions[:, 1:] - positions[:, :-1], axis=2)[horizontal]
        )
    vertical = valid[:-1, :] & valid[1:, :]
    if vertical.any():
        edges.append(np.linalg.norm(positions[1:] - positions[:-1], axis=2)[vertical])
    if not edges:
        return 0.0
    return float(np.median(np.concatenate(edges)))


def _build_spatial_hash(positions, sigma):
    flat = positions.reshape(-1, 3)
    valid = ~np.isnan(flat[:, 0])
    points = flat[valid]
    low = points.min(axis=0)
    high = points.max(axis=0)
    max_extent = float(max(high - low))
    cell = max(3.0 * sigma, max_extent / 256.0)
    origin = low - cell
    grid = np.ceil((high - low) / cell).astype(np.int64) + 2
    total = int(grid[0] * grid[1] * grid[2])
    cell_index = np.floor((points - origin) / cell).astype(np.int64)
    linear = (
        cell_index[:, 0]
        + cell_index[:, 1] * grid[0]
        + cell_index[:, 2] * grid[0] * grid[1]
    )
    counts = np.bincount(linear, minlength=total)
    prefix = np.zeros(total + 1, np.uint32)
    np.cumsum(counts, out=prefix[1:])
    order = np.argsort(linear, kind="stable")
    items = np.flatnonzero(valid)[order].astype(np.uint32)
    return grid.astype(np.int32), origin.astype(np.float32), float(cell), prefix, items


class Optimizer(_OpenCLProgram):
    """The expand, discovery/refinement and blur kernels over the UV maps."""

    def __init__(self, device, texel_count, cap):
        super().__init__(device)
        self.texel_count = texel_count
        self.cap = cap
        optimize_program, optimize_kernel = _build_program(
            self._cl, self._context, device, OPTIMIZER_KERNEL, "optimize"
        )
        expand_program, expand_kernel = _build_program(
            self._cl, self._context, device, EXPAND_KERNEL, "expand"
        )
        blur_program, blur_kernel = _build_program(
            self._cl, self._context, device, BLUR_KERNEL, "blur"
        )
        self._programs = (optimize_program, expand_program, blur_program)
        self._optimize = _Kernel(self._cl, self._queue, optimize_kernel)
        self._expand = _Kernel(self._cl, self._queue, expand_kernel)
        self._blur = _Kernel(self._cl, self._queue, blur_kernel)

        self._positions = self._buffer(texel_count * 3 * 4, CL_MEM_READ_ONLY)
        self._normals = self._buffer(texel_count * 3 * 4, CL_MEM_READ_ONLY)
        self._tangents = self._buffer(texel_count * 3 * 4, CL_MEM_READ_ONLY)
        self._lights = None
        self._lights_size = 0
        self._pool = self._buffer(
            texel_count * cap * WITNESS_WORDS * 4, CL_MEM_READ_ONLY
        )
        self._counts = self._buffer(texel_count * 4, CL_MEM_READ_ONLY)
        self._albedo = self._buffer(texel_count * 16, CL_MEM_READ_WRITE)
        self._roughness = self._buffer(texel_count * 4, CL_MEM_READ_WRITE)
        self._metallic = self._buffer(texel_count * 4, CL_MEM_READ_WRITE)
        self._normal = self._buffer(texel_count * 16, CL_MEM_READ_WRITE)
        self._errors = self._buffer(texel_count * 16, CL_MEM_READ_WRITE)
        self._weights = self._buffer(texel_count * 4, CL_MEM_READ_WRITE)
        self._candidates = self._buffer(20 * 4, CL_MEM_READ_ONLY)
        self._params = self._buffer(9 * 4, CL_MEM_READ_ONLY)
        self._dst_albedo = self._buffer(texel_count * 16, CL_MEM_READ_WRITE)
        self._dst_roughness = self._buffer(texel_count * 4, CL_MEM_READ_WRITE)
        self._dst_metallic = self._buffer(texel_count * 4, CL_MEM_READ_WRITE)
        self._dst_normal = self._buffer(texel_count * 16, CL_MEM_READ_WRITE)
        self._cell_start = self._buffer(4, CL_MEM_READ_ONLY)
        self._cell_items = self._buffer(texel_count * 4, CL_MEM_READ_ONLY)
        self._cell_capacity = 0
        self._expanded = None
        self._tile_capacity = 0
        self.reset_maps()

    def blur(self, radius, positions, normals, tangents):
        sigma = radius * _world_texel_size(positions)
        if not np.isfinite(sigma) or sigma <= 0.0:
            return False
        grid, origin, cell, prefix, items = _build_spatial_hash(positions, sigma)
        if self._cell_capacity < prefix.size:
            if self._cell_capacity:
                self._cl.clReleaseMemObject(self._cell_start)
            self._cell_start = self._buffer(prefix.nbytes, CL_MEM_READ_ONLY)
            self._cell_capacity = prefix.size
        self._write(self._cell_start, prefix.tobytes())
        self._write(self._cell_items, items.tobytes())
        self._cl.clFinish(self._queue)

        kernel = self._blur
        for index, memory in enumerate(
            (
                self._positions,
                self._cell_start,
                self._cell_items,
                self._normals,
                self._tangents,
                self._albedo,
                self._roughness,
                self._metallic,
                self._normal,
                self._dst_albedo,
                self._dst_roughness,
                self._dst_metallic,
                self._dst_normal,
            )
        ):
            kernel.set_buffer(index, memory)
        kernel.set_uint(13, self.texel_count)
        kernel.set_int(14, int(grid[0]))
        kernel.set_int(15, int(grid[1]))
        kernel.set_int(16, int(grid[2]))
        kernel.set_float(17, float(origin[0]))
        kernel.set_float(18, float(origin[1]))
        kernel.set_float(19, float(origin[2]))
        kernel.set_float(20, cell)
        kernel.set_float(21, 2.0 * sigma * sigma)
        kernel.set_float(22, 9.0 * sigma * sigma)
        kernel.launch(((self.texel_count + 31) // 32) * 32, local_size=32)
        self._albedo, self._dst_albedo = self._dst_albedo, self._albedo
        self._roughness, self._dst_roughness = self._dst_roughness, self._roughness
        self._metallic, self._dst_metallic = self._dst_metallic, self._metallic
        self._normal, self._dst_normal = self._dst_normal, self._normal
        return True

    def reset_maps(self):
        n = self.texel_count
        albedo = np.zeros((n, 4), np.float32)
        albedo[:, :3] = INITIAL_ALBEDO
        albedo[:, 3] = 1.0
        normal = np.zeros((n, 4), np.float32)
        normal[:, :3] = INITIAL_NORMAL
        normal[:, 3] = 1.0
        self._write(self._albedo, albedo.tobytes())
        self._write(
            self._roughness, np.full(n, INITIAL_ROUGHNESS, np.float32).tobytes()
        )
        self._write(self._metallic, np.full(n, INITIAL_METALLIC, np.float32).tobytes())
        self._write(self._normal, normal.tobytes())
        self._write(self._errors, np.zeros((n, 4), np.float32).tobytes())
        self._cl.clFinish(self._queue)

    def set_scene(self, positions, normals, tangents, lights, pool, counts):
        self._write(self._positions, positions.tobytes())
        self._write(self._normals, normals.tobytes())
        self._write(self._tangents, tangents.tobytes())
        self.set_lights(lights)
        self._write(self._pool, pool.tobytes())
        self._write(self._counts, counts.tobytes())
        self._cl.clFinish(self._queue)

    def set_lights(self, lights):
        if self._lights is None or self._lights_size != lights.nbytes:
            if self._lights is not None:
                self._cl.clReleaseMemObject(self._lights)
            self._lights = self._buffer(lights.nbytes, CL_MEM_READ_ONLY)
            self._lights_size = lights.nbytes
        self._write(self._lights, lights.tobytes())
        self._cl.clFinish(self._queue)

    def prepare_tile(self, tile_pixels):
        if self._expanded is None or self._tile_capacity < tile_pixels:
            if self._expanded is not None:
                self._cl.clReleaseMemObject(self._expanded)
            self._tile_capacity = tile_pixels
            self._expanded = self._buffer(
                tile_pixels * self.cap * 7 * 4, CL_MEM_READ_WRITE
            )

    def expand_tile(self, tile_offset, tile_count, shading_gate):
        kernel = self._expand
        for index, memory in enumerate(
            (
                self._pool,
                self._counts,
                self._positions,
                self._normals,
                self._lights,
                self._expanded,
                self._weights,
            )
        ):
            kernel.set_buffer(index, memory)
        kernel.set_uint(7, tile_offset)
        kernel.set_uint(8, tile_count)
        kernel.set_uint(9, self.cap)
        kernel.set_float(10, shading_gate)
        kernel.launch(((tile_count + 63) // 64) * 64)

    def set_candidates(self, candidates):
        self._write(self._candidates, np.asarray(candidates, np.float32).tobytes())
        self._cl.clFinish(self._queue)

    def set_params(
        self,
        ambient,
        start_jitter,
        step_alpha,
        min_step,
        grad_tol,
        loss_tol,
        rms_decay,
        rms_eps,
        use_long_bb,
    ):
        params = np.array(
            [
                ambient,
                start_jitter,
                step_alpha,
                min_step,
                grad_tol,
                loss_tol,
                rms_decay,
                rms_eps,
                1.0 if use_long_bb else 0.0,
            ],
            np.float32,
        )
        self._write(self._params, params.tobytes())
        self._cl.clFinish(self._queue)

    def optimize_tile(
        self,
        tile_offset,
        pixel_count,
        estimate_albedo,
        use_candidates,
        subpass,
        candidate_count,
        max_iters,
    ):
        kernel = self._optimize
        for index, memory in enumerate(
            (
                self._positions,
                self._normals,
                self._tangents,
                self._lights,
                self._expanded,
                self._counts,
                self._albedo,
                self._roughness,
                self._metallic,
                self._normal,
                self._errors,
                self._candidates,
                self._params,
            )
        ):
            kernel.set_buffer(index, memory)
        kernel.set_uint(13, pixel_count)
        kernel.set_uint(14, self.cap)
        kernel.set_uint(15, tile_offset)
        kernel.set_int(16, 1 if estimate_albedo else 0)
        kernel.set_int(17, 1 if use_candidates else 0)
        kernel.set_uint(18, subpass)
        kernel.set_uint(19, candidate_count)
        kernel.set_int(20, max_iters)
        kernel.launch(((pixel_count + 63) // 64) * 64)

    def read_maps(self):
        n = self.texel_count
        albedo = self._read(self._albedo, n * 16, np.float32).reshape(n, 4)
        roughness = self._read(self._roughness, n * 4, np.float32)
        metallic = self._read(self._metallic, n * 4, np.float32)
        normal = self._read(self._normal, n * 16, np.float32).reshape(n, 4)
        errors = self._read(self._errors, n * 16, np.float32).reshape(n, 4)
        return albedo, roughness, metallic, normal, errors

    def read_weights(self):
        return self._read(self._weights, self.texel_count * 4, np.float32)

    def destroy(self):
        self._cl.clFinish(self._queue)
        memories = [
            self._positions,
            self._normals,
            self._tangents,
            self._pool,
            self._counts,
            self._albedo,
            self._roughness,
            self._metallic,
            self._normal,
            self._errors,
            self._weights,
            self._candidates,
            self._params,
            self._dst_albedo,
            self._dst_roughness,
            self._dst_metallic,
            self._dst_normal,
            self._cell_start,
            self._cell_items,
        ]
        if self._lights is not None:
            memories.append(self._lights)
        if self._expanded is not None:
            memories.append(self._expanded)
        for memory in memories:
            self._cl.clReleaseMemObject(memory)
        for kernel in (self._optimize, self._expand, self._blur):
            self._cl.clReleaseKernel(kernel.handle)
        for program in self._programs:
            self._cl.clReleaseProgram(program)


_IMAGE_DTYPES = {
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}
_IMAGE_MAX = {np.uint8: 255.0, np.uint16: 65535.0, np.uint32: 4294967295.0}


def _image_to_array(image):
    dtype = _IMAGE_DTYPES.get(image.data_type, np.uint8)
    pixels = image.width * image.height
    data = np.frombuffer(image.tostring(), dtype=dtype)
    channels = data.size // pixels if pixels else 0
    return data.reshape(image.height, image.width, channels)


def _load_photo(camera):
    array = _image_to_array(camera.photo.image())
    maximum = _IMAGE_MAX.get(array.dtype.type, 1.0)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return array[:, :, :3].astype(np.float32) / maximum


def _render_depth(model, camera):
    array = _image_to_array(
        model.renderDepth(camera.transform, camera.sensor.calibration)
    )
    return array[:, :, :2].astype(np.float32)


def _bounds_radius(positions):
    flat = np.asarray(positions).reshape(-1, 3)
    finite = flat[~np.isnan(flat[:, 0])]
    if not len(finite):
        return 1.0
    return float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0))) * 0.5


def _light_position(cam_pos, right, up, forward, light_offset):
    return (
        cam_pos
        + right * light_offset[0]
        + up * light_offset[1]
        + forward * light_offset[2]
    )


def _image_lights(cameras, light_multiplier, light_offset, radius):
    intensity = light_multiplier * radius * radius
    image_lights = np.zeros((len(cameras), 7), np.float32)
    for index, camera in enumerate(cameras):
        cam_pos, right, up, forward = _camera_axes(camera)
        light_pos = _light_position(cam_pos, right, up, forward, light_offset)
        image_lights[index, 0:3] = cam_pos
        image_lights[index, 3:6] = light_pos
        image_lights[index, 6] = intensity
    return image_lights


def _gather(
    model,
    positions,
    normals,
    cameras,
    size,
    device,
    light_multiplier,
    light_offset,
    max_witnesses,
    report=None,
    occlusion_radius=GATHER_OCCLUSION_RADIUS,
    shading_gate=GATHER_SHADING_GATE,
    depth_eps_abs=GATHER_DEPTH_EPS_ABS,
    depth_eps_rel=GATHER_DEPTH_EPS_REL,
    border_fade=128.0,
    cancel=None,
):
    pixel_count = size * size
    if not cameras:
        return None, None
    flat_positions = positions.reshape(-1, 3).astype(np.float32)
    flat_normals = normals.reshape(-1, 3).astype(np.float32)
    radius = _bounds_radius(flat_positions)
    intensity = light_multiplier * radius * radius
    print(
        f"Light: radius={radius:.4f}, multiplier={light_multiplier:.3f}, intensity={intensity:.4f}",
        flush=True,
    )
    image_lights = np.zeros((len(cameras), 7), np.float32)
    gatherer = WitnessGatherer(device, pixel_count, max_witnesses)
    try:
        gatherer.set_geometry(flat_positions.tobytes(), flat_normals.tobytes())
        for camera_number, camera in enumerate(cameras, start=1):
            _check_cancel(cancel)
            started = time.time()
            label = camera.label or f"Camera {camera.key}"
            if report is not None:
                report(
                    (camera_number - 1) / len(cameras),
                    f"Gathering witnesses: {label}",
                )
            calibration = camera.sensor.calibration
            cam_w, cam_h = int(calibration.width), int(calibration.height)
            lut, lut_grid, lut_smin, lut_sspan, lut_tmin, lut_tspan = _projection_lut(
                camera
            )
            depth = _render_depth(model, camera)
            photo = _load_photo(camera)
            if photo.shape[0] != cam_h or photo.shape[1] != cam_w:
                print(
                    f"Gather {camera_number}/{len(cameras)}: {label}: photo {photo.shape[1]}x{photo.shape[0]} != calibration {cam_w}x{cam_h}, skipping",
                    flush=True,
                )
                continue
            cam_pos, right, up, forward = _camera_axes(camera)
            light_pos = _light_position(cam_pos, right, up, forward, light_offset)
            image_lights[camera_number - 1, 0:3] = cam_pos
            image_lights[camera_number - 1, 3:6] = light_pos
            image_lights[camera_number - 1, 6] = intensity
            params = np.array(
                [
                    cam_pos[0],
                    cam_pos[1],
                    cam_pos[2],
                    right[0],
                    right[1],
                    right[2],
                    -up[0],
                    -up[1],
                    -up[2],
                    forward[0],
                    forward[1],
                    forward[2],
                    light_pos[0],
                    light_pos[1],
                    light_pos[2],
                    intensity,
                    lut_smin,
                    lut_sspan,
                    lut_tmin,
                    lut_tspan,
                    depth_eps_abs,
                    depth_eps_rel,
                    shading_gate,
                    float(border_fade),
                    GATHER_OCCLUSION_NEAR_FADE,
                    GATHER_OCCLUSION_VALID_FADE,
                ],
                np.float32,
            )
            gatherer.run_camera(
                lut,
                lut_grid,
                depth,
                cam_w,
                cam_h,
                photo,
                params,
                camera_number - 1,
                occlusion_radius,
            )
            print(
                f"Gather {camera_number}/{len(cameras)}: {label} ({time.time() - started:.2f}s)",
                flush=True,
            )
        if report is not None:
            report(1.0, "Gathering witnesses")
        coverage = gatherer.coverage().reshape(size, size)
        pool = gatherer.read_pool()
        counts = np.minimum(gatherer.read_counts(), np.uint32(max_witnesses))
        return coverage, image_lights, pool, counts
    finally:
        gatherer.destroy()


def _save_coverage(out_dir, coverage):
    _save_array_png(
        os.path.join(out_dir, "witness_coverage.png"),
        np.repeat(coverage[..., None], 3, axis=2),
    )
    print("Witness coverage saved to: " + out_dir)


def _run_pass(
    optimizer,
    texel_count,
    tile_pixels,
    dialog,
    candidate_count,
    discovery,
    subpasses,
    on_dispatch=None,
    cancel=None,
):
    """Expands and optimizes every tile of one pass; returns (dispatches, tiles)."""
    optimizer.prepare_tile(tile_pixels)
    tiles = (texel_count + tile_pixels - 1) // tile_pixels
    total = tiles * subpasses
    dispatched = 0
    for tile_start in range(0, texel_count, tile_pixels):
        _check_cancel(cancel)
        count = min(tile_pixels, texel_count - tile_start)
        optimizer.expand_tile(tile_start, count, dialog.shading_gate)
        for subpass in range(subpasses):
            _check_cancel(cancel)
            optimizer.optimize_tile(
                tile_start,
                count,
                discovery,
                discovery,
                subpass,
                candidate_count,
                dialog.max_iterations,
            )
            dispatched += 1
            if on_dispatch is not None:
                on_dispatch(dispatched, total, tile_start, count, subpass)
    return dispatched, tiles


def _run_build(
    positions,
    normals,
    tangents,
    image_lights,
    pool,
    counts,
    size,
    device,
    dialog,
    report,
    cancel=None,
):
    texel_count = size * size
    candidates = dialog.roughness_candidates[:20] or [0.5]
    tile_pixels = min(texel_count, max(1024, dialog.tile_pixels))
    started = time.time()
    optimizer = Optimizer(device, texel_count, dialog.max_witnesses)
    try:
        optimizer.set_scene(
            positions.reshape(-1, 3).astype(np.float32),
            normals.reshape(-1, 3).astype(np.float32),
            tangents.reshape(-1, 3).astype(np.float32),
            image_lights.astype(np.float32),
            pool,
            counts.astype(np.uint32),
        )
        optimizer.set_candidates(candidates)
        optimizer.set_params(
            dialog.ambient,
            dialog.start_jitter,
            dialog.step_alpha,
            dialog.min_step,
            dialog.grad_tol,
            dialog.loss_tol,
            dialog.rms_decay,
            dialog.rms_eps,
            dialog.use_long_bb,
        )
        print(
            f"Discovery: {texel_count} texels, {len(candidates)} candidate(s), {dialog.max_iterations} iters, {dialog.subpasses} subpass(es), "
            f"tile={tile_pixels}",
            flush=True,
        )

        def discovery_progress(dispatched, total, tile_start, count, subpass):
            report(0.55 + 0.30 * dispatched / max(1, total), "Discovery pass")
            print(
                f"Discovery dispatch {dispatched}/{total}: tile {tile_start // tile_pixels + 1}, subpass {subpass + 1} ({count} texels)",
                flush=True,
            )

        dispatched, _ = _run_pass(
            optimizer,
            texel_count,
            tile_pixels,
            dialog,
            len(candidates),
            True,
            dialog.subpasses,
            discovery_progress,
            cancel=cancel,
        )
        print(
            f"Discovery finished in {time.time() - started:.1f}s ({dispatched} dispatches)",
            flush=True,
        )

        if not dialog.discovery_only:
            if optimizer.blur(dialog.blur_radius, positions, normals, tangents):
                report(0.88, "Blur pass")
                print("Blur pass done", flush=True)

            def refinement_progress(dispatched, total, tile_start, count, subpass):
                report(0.88 + 0.07 * dispatched / max(1, total), "Refinement pass")
                print(
                    f"Refinement dispatch {dispatched}/{total}: tile {tile_start // tile_pixels + 1} ({count} texels)",
                    flush=True,
                )

            _run_pass(
                optimizer,
                texel_count,
                tile_pixels,
                dialog,
                1,
                False,
                1,
                refinement_progress,
                cancel=cancel,
            )
        print(f"Build finished in {time.time() - started:.1f}s", flush=True)
        return optimizer.read_maps()
    finally:
        optimizer.destroy()


def _save_gray_png(path, image):
    _save_array_png(path, np.repeat(np.asarray(image)[..., None], 3, axis=2))


def _save_discovery_maps(
    out_dir, size, albedo, roughness, metallic, normal_map, errors
):
    _save_array_png(
        os.path.join(out_dir, "albedo.png"), albedo[:, :3].reshape(size, size, 3)
    )
    _save_gray_png(
        os.path.join(out_dir, "roughness.png"), roughness.reshape(size, size)
    )
    _save_gray_png(os.path.join(out_dir, "metallic.png"), metallic.reshape(size, size))
    _save_array_png(
        os.path.join(out_dir, "normal.png"), normal_map[:, :3].reshape(size, size, 3)
    )
    _save_array_png(
        os.path.join(out_dir, "photo_errors_x5.png"),
        (errors[:, :3] * 5.0).reshape(size, size, 3),
    )
    print("PBR textures saved to: " + out_dir)


FIT_RES_COARSE = 256
FIT_RES_REFINE = 512
COVERAGE_MIN = 0.05
IMPROVE_EPS = 0.005


def _albedo_stats(albedo, coverage):
    values = albedo[coverage >= COVERAGE_MIN, :3]
    if not values.shape[0]:
        return 0.0, 0.0, 0.0
    lightness = values.max(axis=1)
    return (
        float(lightness.mean()),
        float((lightness >= 0.99).mean()),
        float((lightness <= 0.01).mean()),
    )


def _multiplier_acceptable(stats):
    mean, clip_hi, clip_lo = stats
    return clip_hi + clip_lo < 0.25 and mean >= 0.10


def _mean_photo_error(errors, coverage):
    values = errors[coverage >= COVERAGE_MIN, :3]
    if not values.shape[0]:
        return 1.0
    return float(values.mean())


def _weighted_photo_error(errors, weights):
    total = float(weights.sum())
    if total <= 1e-9:
        return 1.0, 0.0
    channel = np.nan_to_num(errors[:, :3].mean(axis=1), nan=0.0)
    return float((weights * channel).sum() / total), total


def _fit_score(score, baseline_weight):
    error, weight = score
    ratio = weight / max(baseline_weight, 1e-9)
    excess = max(0.0, 1.0 - ratio - 0.2)
    return error * (1.0 + 2.0 * excess)


def _normal_deviation(normal_map, geom_normals, tangents, coverage):
    cos30 = 0.8660254
    geom = geom_normals.reshape(-1, 3)
    tangent = tangents.reshape(-1, 3)
    geom_len = np.linalg.norm(geom, axis=1)
    tangent_len = np.linalg.norm(tangent, axis=1)
    valid = (coverage > 0.0) & (geom_len > 1e-6) & (tangent_len > 1e-6)
    if not valid.any():
        return 0.0
    gn = geom[valid] / geom_len[valid, None]
    tg = tangent[valid] / tangent_len[valid, None]
    bitangent = np.cross(gn, tg)
    bitangent_len = np.linalg.norm(bitangent, axis=1, keepdims=True)
    bitangent_len[bitangent_len < 1e-8] = 1.0
    bitangent = bitangent / bitangent_len
    decoded = normal_map[valid, :3] * 2.0 - 1.0
    decoded_len = np.linalg.norm(decoded, axis=1, keepdims=True)
    decoded_len[decoded_len < 1e-8] = 1.0
    decoded = decoded / decoded_len
    world = tg * decoded[:, 0:1] + bitangent * decoded[:, 1:2] + gn * decoded[:, 2:3]
    world_len = np.linalg.norm(world, axis=1, keepdims=True)
    world_len[world_len < 1e-8] = 1.0
    world = world / world_len
    return float(((world * gn).sum(axis=1) < cos30).mean())


class _LightFitter:
    def __init__(self, model, cameras, device, dialog, report=None, cancel=None):
        self.model = model
        self.cameras = list(cameras)
        self.device = device
        self.dialog = dialog
        self.report = report or (lambda fraction, text=None: None)
        self.cancel = cancel
        self.multiplier = float(dialog.light_multiplier)
        self.offset = tuple(float(value) for value in dialog.light_offset)
        self.baseline_error = 1.0
        self.baseline_weight = 0.0
        self.best_offset = (0.0, 0.0, 0.0)
        self.best_score = (1.0, 0.0)
        self.z_max = 0.0
        self.tested = []
        self.candidates = dialog.roughness_candidates[:20] or [0.5]
        self.optimizer = None
        self.res = 0
        self.radius = 1.0
        self.positions = self.normals = self.tangents = None
        self.last = None

    def _clamp_z(self, offset):
        return (offset[0], offset[1], min(max(offset[2], -self.z_max), self.z_max))

    def _setup(self, res):
        if self.optimizer is not None:
            self.optimizer.destroy()
            self.optimizer = None
        self.res = res
        self.positions, self.normals, self.tangents = _bake_geometry(
            self.model, res, cancel=self.cancel
        )
        self.radius = _bounds_radius(self.positions)
        _, _, pool, counts = _gather(
            self.model,
            self.positions,
            self.normals,
            self.cameras,
            res,
            self.device,
            self.multiplier,
            self.offset,
            self.dialog.max_witnesses,
            self.report,
            self.dialog.occlusion_radius,
            self.dialog.shading_gate,
            self.dialog.depth_eps_abs,
            self.dialog.depth_eps_rel,
            float(self.dialog.no_data_fade),
            cancel=self.cancel,
        )
        self.optimizer = Optimizer(self.device, res * res, self.dialog.max_witnesses)
        self.optimizer.set_scene(
            self.positions.reshape(-1, 3).astype(np.float32),
            self.normals.reshape(-1, 3).astype(np.float32),
            self.tangents.reshape(-1, 3).astype(np.float32),
            _image_lights(self.cameras, self.multiplier, self.offset, self.radius),
            pool,
            counts.astype(np.uint32),
        )
        self.optimizer.set_candidates(self.candidates)
        self.optimizer.set_params(
            self.dialog.ambient,
            self.dialog.start_jitter,
            self.dialog.step_alpha,
            self.dialog.min_step,
            self.dialog.grad_tol,
            self.dialog.loss_tol,
            self.dialog.rms_decay,
            self.dialog.rms_eps,
            self.dialog.use_long_bb,
        )

    def _build(self, offset, fraction):
        """Runs a discovery-only build at the fit resolution for `offset`."""
        self.optimizer.set_lights(
            _image_lights(self.cameras, self.multiplier, offset, self.radius)
        )
        self.optimizer.reset_maps()
        texel_count = self.res * self.res
        tile_pixels = min(texel_count, max(1024, self.dialog.tile_pixels))
        _run_pass(
            self.optimizer,
            texel_count,
            tile_pixels,
            self.dialog,
            len(self.candidates),
            True,
            self.dialog.subpasses,
            cancel=self.cancel,
        )
        albedo, _roughness, _metallic, normal_map, errors = self.optimizer.read_maps()
        weights = self.optimizer.read_weights()
        maximum = float(weights.max()) if weights.size else 0.0
        coverage = weights / maximum if maximum > 0.0 else weights
        normal_dev = _normal_deviation(
            normal_map, self.normals, self.tangents, coverage
        )
        self.last = (albedo, normal_map, errors, coverage, weights, normal_dev)
        self.report(fraction, "Fitting light")
        return _weighted_photo_error(errors, weights)

    def _test(self, offset, fraction):
        offset = self._clamp_z(offset)
        for tested, score in self.tested:
            if np.linalg.norm(np.asarray(tested) - np.asarray(offset)) < 1e-4:
                return score
        score = self._build(offset, fraction)
        self.tested.append((offset, score))
        print(
            f"Light offset candidate ({offset[0]:.4f},{offset[1]:.4f},{offset[2]:.4f}): "
            f"error={score[0]:.5f} weight={score[1]:.1f} normal_dev={self.last[5]:.4f}",
            flush=True,
        )
        return score

    def _beats(self, candidate):
        return _fit_score(candidate, self.baseline_weight) < _fit_score(
            self.best_score, self.baseline_weight
        ) * (1.0 - IMPROVE_EPS)

    def _beats_fine(self, candidate):
        return _fit_score(candidate, self.baseline_weight) < _fit_score(
            self.best_score, self.baseline_weight
        ) * (1.0 - 1e-4)

    def _fit_multiplier(self, fraction0, fraction1):
        seed_multiplier = self.multiplier
        self.offset = (0.0, 0.0, 0.0)
        factors = (0.125, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
        span = fraction1 - fraction0
        results = []
        for index, factor in enumerate(factors):
            self.multiplier = seed_multiplier * factor
            error, _ = self._build(
                (0.0, 0.0, 0.0), fraction0 + span * (index + 1) / (len(factors) + 1)
            )
            stats = _albedo_stats(self.last[0], self.last[3])
            normal_dev = self.last[5]
            results.append((self.multiplier, error, stats))
            print(
                f"Light multiplier candidate {self.multiplier:.3f}: mean={stats[0]:.3f} clip_hi={stats[1]:.3f} "
                f"clip_lo={stats[2]:.3f} error={error:.5f} normal_dev={normal_dev:.4f}",
                flush=True,
            )

        acceptable = [result for result in results if _multiplier_acceptable(result[2])]
        pool = acceptable or results
        best_error = min(result[1] for result in pool)
        near_best = [result for result in pool if result[1] <= best_error * 1.05 + 1e-6]
        best_multiplier, _, best_stats = max(near_best, key=lambda result: result[2][0])
        self.multiplier = best_multiplier
        self._build((0.0, 0.0, 0.0), fraction1)
        self.baseline_error = _mean_photo_error(self.last[2], self.last[3])
        print(
            f"Light multiplier fitted to {self.multiplier:.3f} (mean={best_stats[0]:.3f} clip_hi={best_stats[1]:.3f} "
            f"baseline error {self.baseline_error:.4f})",
            flush=True,
        )

    def _fit_offset(self, fraction0, fraction1):
        radius = max(self.radius, 1e-3)
        m1 = 0.2 * radius
        self.offset = (0.0, 0.0, 0.0)
        base_fraction = fraction0 + (fraction1 - fraction0) * 0.05
        baseline = self._build((0.0, 0.0, 0.0), base_fraction)
        self.baseline_weight = baseline[1]
        self.best_offset = (0.0, 0.0, 0.0)
        self.best_score = baseline
        self.tested = [((0.0, 0.0, 0.0), baseline)]
        probe_fraction = fraction0 + (fraction1 - fraction0) * 0.30
        probe_end = fraction0 + (fraction1 - fraction0) * 0.55
        probe_count = 0
        for magnitude in (
            0.01 * radius,
            0.02 * radius,
            0.05 * radius,
            0.1 * radius,
            0.2 * radius,
            0.5 * radius,
        ):
            for direction in (
                (magnitude, 0.0, 0.0),
                (-magnitude, 0.0, 0.0),
                (0.0, magnitude, 0.0),
                (0.0, -magnitude, 0.0),
            ):
                fraction = (
                    probe_fraction + (probe_end - probe_fraction) * probe_count / 12.0
                )
                candidate = self._test(direction, fraction)
                if self._beats(candidate):
                    self.best_score = candidate
                    self.best_offset = self._clamp_z(direction)
                probe_count += 1

        if np.linalg.norm(np.asarray(self.best_offset)) > 1e-6:
            refine_fraction = fraction1 - 0.10
            step = 0.3 * m1
            for _ in range(8):
                improved = False
                for axis in range(2):
                    for sign in (1.0, -1.0):
                        offset = list(self.best_offset)
                        offset[axis] += sign * step
                        candidate = self._test(tuple(offset), refine_fraction)
                        if self._beats_fine(candidate):
                            self.best_score = candidate
                            self.best_offset = self._clamp_z(tuple(offset))
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 0.01 * m1:
                        break

        self.offset = self.best_offset
        print(
            f"Light offset fitted to ({self.offset[0]:.4f},{self.offset[1]:.4f},{self.offset[2]:.4f}) (error {self.best_score[0]:.4f}, weight {self.best_score[1]:.2f})",
            flush=True,
        )

    def run(self):
        try:
            self._setup(FIT_RES_COARSE)
            self._fit_multiplier(0.0, 0.25)
            self._setup(FIT_RES_REFINE)
            self._fit_offset(0.25, 1.0)
            return self.multiplier, self.offset, self.best_score[0]
        finally:
            if self.optimizer is not None:
                self.optimizer.destroy()
                self.optimizer = None


class ProgressDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("PBR Builder")
        self.setModal(True)
        self.token = _CancelToken()
        self._done = False

        self.status = QtWidgets.QLabel("Starting...")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumSize(560, 240)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def report(self, fraction, message=None):
        self.progress.setValue(int(np.clip(fraction, 0.0, 1.0) * 100))
        if message:
            self.status.setText(message)
        self._pump()

    def append(self, line):
        if line:
            self.log.appendPlainText(line)
        self._pump()

    def finish(self):
        self._done = True
        self.progress.setValue(100)
        self.status.setText("Cancelled" if self.token.cancelled else "Finished")
        self._pump()

    def closeEvent(self, event):
        if not self._done:
            self._on_cancel()
        super().closeEvent(event)

    def _on_cancel(self):
        if self._done or self.token.cancelled:
            return
        self.status.setText("Cancelling...")
        self.cancel_button.setEnabled(False)
        self.token.cancel()

    def _pump(self):
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)


class _ConsoleTee:
    def __init__(self, stream, callback):
        self._stream = stream
        self._callback = callback
        self._buffer = ""

    def write(self, text):
        if self._stream is not None:
            self._stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._callback(line)

    def flush(self):
        if self._stream is not None:
            self._stream.flush()


@contextlib.contextmanager
def _progress_session(parent):
    """Shows a modal progress dialog and mirrors stdout into its log for the block."""
    progress = ProgressDialog(parent)
    progress.show()
    original_stdout = sys.stdout
    sys.stdout = _ConsoleTee(original_stdout, progress.append)
    try:
        yield progress
    finally:
        progress.finish()
        progress.close()
        sys.stdout = original_stdout


def _power_combo(values, default):
    combo = QtWidgets.QComboBox()
    for value in values:
        combo.addItem(str(value), value)
    index = combo.findData(default)
    combo.setCurrentIndex(max(index, 0))
    return combo


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setArrowType(QtCore.Qt.RightArrow)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.content = QtWidgets.QWidget()
        self.content.setVisible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        self.toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )
        self.content.setVisible(checked)

    def setContentLayout(self, content_layout):
        self.content.setLayout(content_layout)


class BakeDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent,
        model_labels,
        model_has_uv,
        device_names,
        camera_labels,
        default_size,
        default_model_index=0,
        default_light=10.0,
        light_settings=None,
        settings=None,
    ):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Build PBR textures")
        self._model_has_uv = model_has_uv
        self._default_light = default_light

        self.model_cmb = QtWidgets.QComboBox()
        self.model_cmb.addItems(model_labels)
        self.model_cmb.setCurrentIndex(default_model_index)
        self.device_cmb = QtWidgets.QComboBox()
        self.device_cmb.addItems(device_names)
        self.size_cmb = _power_combo(
            [64, 128, 256, 512, 1024, 2048, 4096, 8192], default_size
        )

        self.keep_uv_radio = QtWidgets.QRadioButton("Keep existing UVs")
        self.rebuild_uv_radio = QtWidgets.QRadioButton("Rebuild UVs")
        self.uv_mode_group = QtWidgets.QButtonGroup(self)
        self.uv_mode_group.addButton(self.keep_uv_radio)
        self.uv_mode_group.addButton(self.rebuild_uv_radio)
        self.mapping_cmb = QtWidgets.QComboBox()
        for name, _ in MAPPING_MODES:
            self.mapping_cmb.addItem(name)
        self.page_spin = QtWidgets.QSpinBox()
        self.page_spin.setRange(1, 64)
        self.texture_size_cmb = _power_combo(
            [1024, 2048, 4096, 8192, 16384, 32768], 8192
        )
        self.pixel_size_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.0, 1000.0)
        self.pixel_size_spin.setDecimals(6)
        self.pixel_size_spin.setToolTip("0 uses the automatic texture resolution")
        self.camera_cmb = QtWidgets.QComboBox()
        for label, key in camera_labels:
            self.camera_cmb.addItem(label, key)
        self.export_baked_check = QtWidgets.QCheckBox(
            "Save baked maps (position, normal, tangent)"
        )
        self.export_baked_check.setChecked(False)
        self.light_spin = QtWidgets.QDoubleSpinBox()
        self.light_spin.setRange(0.0, 1000.0)
        self.light_spin.setDecimals(3)
        self.light_spin.setValue(default_light)
        self.witness_cmb = _power_combo(
            [1, 2, 4, 8, 10, 12, 16, 20, 25, 32, 40, 48, 64], 32
        )
        self.occlusion_radius_spin = QtWidgets.QSpinBox()
        self.occlusion_radius_spin.setRange(1, 64)
        self.occlusion_radius_spin.setValue(16)
        self.shading_gate_spin = QtWidgets.QDoubleSpinBox()
        self.shading_gate_spin.setRange(0.0, 0.5)
        self.shading_gate_spin.setDecimals(4)
        self.shading_gate_spin.setValue(0.05)
        self.depth_eps_abs_spin = QtWidgets.QDoubleSpinBox()
        self.depth_eps_abs_spin.setRange(0.0, 1.0)
        self.depth_eps_abs_spin.setDecimals(4)
        self.depth_eps_abs_spin.setValue(0.05)
        self.depth_eps_rel_spin = QtWidgets.QDoubleSpinBox()
        self.depth_eps_rel_spin.setRange(0.0, 0.1)
        self.depth_eps_rel_spin.setDecimals(5)
        self.depth_eps_rel_spin.setValue(0.005)
        self.no_data_fade_spin = QtWidgets.QSpinBox()
        self.no_data_fade_spin.setRange(1, 1024)
        self.no_data_fade_spin.setValue(128)
        self.blur_radius_spin = QtWidgets.QDoubleSpinBox()
        self.blur_radius_spin.setRange(0.0, 50.0)
        self.blur_radius_spin.setDecimals(3)
        self.blur_radius_spin.setValue(1.0)
        self.discovery_only_check = QtWidgets.QCheckBox(
            "Discovery only (skip blur and refinement)"
        )
        self.discovery_only_check.setChecked(False)
        self.fit_light_button = QtWidgets.QPushButton("Fit light")
        self.fit_light_button.setToolTip(
            "Run low-resolution discovery-only test builds to calibrate the light "
            "multiplier and offset, and update the fields above. Does not start the build."
        )
        self.ambient_spin = QtWidgets.QDoubleSpinBox()
        self.ambient_spin.setRange(0.0, 1.0)
        self.ambient_spin.setDecimals(4)
        self.subpasses_spin = QtWidgets.QSpinBox()
        self.subpasses_spin.setRange(1, 64)
        self.subpasses_spin.setValue(5)
        self.jitter_spin = QtWidgets.QDoubleSpinBox()
        self.jitter_spin.setRange(0.0, 1.0)
        self.jitter_spin.setDecimals(4)
        self.jitter_spin.setValue(0.1)
        self.max_iters_spin = QtWidgets.QSpinBox()
        self.max_iters_spin.setRange(1, 100000)
        self.max_iters_spin.setValue(1000)
        self.step_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.step_alpha_spin.setRange(0.0, 10.0)
        self.step_alpha_spin.setDecimals(6)
        self.step_alpha_spin.setValue(0.01)
        self.min_step_spin = QtWidgets.QDoubleSpinBox()
        self.min_step_spin.setRange(0.0, 1.0)
        self.min_step_spin.setDecimals(9)
        self.min_step_spin.setValue(1e-6)
        self.grad_tol_spin = QtWidgets.QDoubleSpinBox()
        self.grad_tol_spin.setRange(0.0, 1.0)
        self.grad_tol_spin.setDecimals(9)
        self.grad_tol_spin.setValue(1e-4)
        self.loss_tol_spin = QtWidgets.QDoubleSpinBox()
        self.loss_tol_spin.setRange(0.0, 1.0)
        self.loss_tol_spin.setDecimals(9)
        self.loss_tol_spin.setValue(1e-5)
        self.rms_decay_spin = QtWidgets.QDoubleSpinBox()
        self.rms_decay_spin.setRange(0.0, 1.0)
        self.rms_decay_spin.setDecimals(4)
        self.rms_decay_spin.setValue(0.9)
        self.rms_eps_spin = QtWidgets.QDoubleSpinBox()
        self.rms_eps_spin.setRange(0.0, 1.0)
        self.rms_eps_spin.setDecimals(12)
        self.rms_eps_spin.setValue(1e-8)
        self.long_bb_check = QtWidgets.QCheckBox("Barzilai-Borwein (long step)")
        self.long_bb_check.setChecked(True)
        self.candidates_edit = QtWidgets.QLineEdit(
            "0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
        )
        self.tile_cmb = _power_combo(
            [65536, 131072, 262144, 524288, 1048576, 2097152, 4194304], 262144
        )
        self.light_x_spin = QtWidgets.QDoubleSpinBox()
        self.light_y_spin = QtWidgets.QDoubleSpinBox()
        self.light_z_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.light_x_spin, self.light_y_spin, self.light_z_spin):
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.05)
        self._sync_uv_mode()
        self._defaults = self.collect_settings()

        if light_settings is not None:
            self.light_spin.setValue(light_settings["multiplier"])
            self.light_x_spin.setValue(light_settings["offset"][0])
            self.light_y_spin.setValue(light_settings["offset"][1])
            self.light_z_spin.setValue(light_settings["offset"][2])
            self.ambient_spin.setValue(light_settings["ambient"])
        if settings is not None:
            self.apply_settings(settings)

        self.model_cmb.currentIndexChanged.connect(self._sync_uv_mode)
        self.rebuild_uv_radio.toggled.connect(self._sync_enabled)
        self.mapping_cmb.currentIndexChanged.connect(self._sync_enabled)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset_button = buttons.addButton(
            "Reset to defaults", QtWidgets.QDialogButtonBox.ResetRole
        )
        reset_button.clicked.connect(self.reset_defaults)

        uv_form = QtWidgets.QFormLayout()
        self._uv_form = uv_form
        uv_form.addRow(self.keep_uv_radio)
        uv_form.addRow(self.rebuild_uv_radio)
        uv_form.addRow("Mapping mode:", self.mapping_cmb)
        uv_form.addRow("Page count:", self.page_spin)
        uv_form.addRow("Texture size:", self.texture_size_cmb)
        uv_form.addRow("Pixel size:", self.pixel_size_spin)
        uv_form.addRow("Camera:", self.camera_cmb)
        self._sync_enabled()

        dataset_form = QtWidgets.QFormLayout()
        dataset_form.addRow("Light multiplier:", self.light_spin)
        dataset_form.addRow("Light offset X:", self.light_x_spin)
        dataset_form.addRow("Light offset Y:", self.light_y_spin)
        dataset_form.addRow("Light offset Z:", self.light_z_spin)
        dataset_form.addRow("Ambient:", self.ambient_spin)
        dataset_form.addRow(self.fit_light_button)

        witness_form = QtWidgets.QFormLayout()
        witness_form.addRow("Max witnesses per texel:", self.witness_cmb)
        witness_form.addRow("Occlusion radius (px):", self.occlusion_radius_spin)
        witness_form.addRow("Shading gate:", self.shading_gate_spin)
        witness_form.addRow("Depth eps abs:", self.depth_eps_abs_spin)
        witness_form.addRow("Depth eps rel:", self.depth_eps_rel_spin)
        witness_form.addRow("No-data fade (px):", self.no_data_fade_spin)

        optimization_form = QtWidgets.QFormLayout()
        optimization_form.addRow("Max iterations:", self.max_iters_spin)
        optimization_form.addRow("Step alpha:", self.step_alpha_spin)
        optimization_form.addRow("Discovery iterations:", self.subpasses_spin)
        optimization_form.addRow("Discovery start jitter:", self.jitter_spin)
        optimization_form.addRow("Roughness candidates:", self.candidates_edit)

        advanced_form = QtWidgets.QFormLayout()
        advanced_form.addRow("Min step:", self.min_step_spin)
        advanced_form.addRow("Grad tol:", self.grad_tol_spin)
        advanced_form.addRow("Loss tol:", self.loss_tol_spin)
        advanced_form.addRow("RMS decay:", self.rms_decay_spin)
        advanced_form.addRow("RMS eps:", self.rms_eps_spin)
        advanced_form.addRow(self.long_bb_check)
        advanced_form.addRow("Tile texels:", self.tile_cmb)

        witness_box = CollapsibleBox("Witness gathering")
        witness_box.setContentLayout(witness_form)
        optimization_box = CollapsibleBox("Optimization passes")
        optimization_box.setContentLayout(optimization_form)
        advanced_box = CollapsibleBox("Advanced optimizer")
        advanced_box.setContentLayout(advanced_form)

        blur_form = QtWidgets.QFormLayout()
        blur_form.addRow("Blur radius:", self.blur_radius_spin)

        build_layout = QtWidgets.QVBoxLayout()
        resolution_form = QtWidgets.QFormLayout()
        resolution_form.addRow("Resolution:", self.size_cmb)
        build_layout.addLayout(resolution_form)
        build_layout.addWidget(witness_box)
        build_layout.addWidget(optimization_box)
        build_layout.addWidget(advanced_box)
        build_layout.addLayout(blur_form)
        build_layout.addWidget(self.discovery_only_check)

        exporting_form = QtWidgets.QFormLayout()
        exporting_form.addRow(self.export_baked_check)

        form = QtWidgets.QFormLayout()
        form.addRow("Model:", self.model_cmb)
        form.addRow("OpenCL device:", self.device_cmb)

        uv_box = CollapsibleBox("UV mapping")
        uv_box.setContentLayout(uv_form)
        dataset_box = CollapsibleBox("Dataset parameters")
        dataset_box.setContentLayout(dataset_form)
        build_box = CollapsibleBox("Build parameters")
        build_box.setContentLayout(build_layout)
        exporting_box = CollapsibleBox("Exporting")
        exporting_box.setContentLayout(exporting_form)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.addLayout(form)
        content_layout.addWidget(uv_box)
        content_layout.addWidget(dataset_box)
        content_layout.addWidget(build_box)
        content_layout.addWidget(exporting_box)
        content_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(scroll, 1)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.setMinimumSize(440, 520)

    def _sync_uv_mode(self):
        has_uv = bool(self._model_has_uv[self.model_index])
        self.keep_uv_radio.setEnabled(has_uv)
        self.keep_uv_radio.setChecked(has_uv)
        self.rebuild_uv_radio.setChecked(not has_uv)
        self._sync_enabled()

    def mark_has_uv(self):
        self._model_has_uv[self.model_index] = True
        self._sync_uv_mode()

    def _sync_enabled(self):
        rebuild = self.rebuild_uv_radio.isChecked()
        mode = self.mapping_mode
        generic = mode == Metashape.MappingMode.GenericMapping
        projected = mode in (
            Metashape.MappingMode.OrthophotoMapping,
            Metashape.MappingMode.AdaptiveOrthophotoMapping,
            Metashape.MappingMode.SphericalMapping,
        )
        camera_mode = mode == Metashape.MappingMode.CameraMapping
        self._set_row_enabled(self.mapping_cmb, rebuild)
        self._set_row_enabled(self.page_spin, rebuild)
        self._set_row_enabled(self.texture_size_cmb, rebuild and generic)
        self._set_row_enabled(self.pixel_size_spin, rebuild and projected)
        self._set_row_enabled(self.camera_cmb, rebuild and camera_mode)

    def _set_row_enabled(self, widget, enabled):
        widget.setEnabled(enabled)
        form = getattr(self, "_uv_form", None)
        if form is not None:
            label = form.labelForField(widget)
            if label is not None:
                label.setEnabled(enabled)

    def apply_light(self, multiplier, offset):
        self.light_spin.setValue(float(multiplier))
        self.light_x_spin.setValue(float(offset[0]))
        self.light_y_spin.setValue(float(offset[1]))
        self.light_z_spin.setValue(float(offset[2]))

    def collect_settings(self):
        return {
            "size": self.size,
            "rebuild_uv": self.rebuild_uv,
            "mapping_mode": self.mapping_cmb.currentIndex(),
            "page_count": self.page_count,
            "texture_size": self.texture_size,
            "pixel_size": self.pixel_size,
            "camera_key": self.camera_key,
            "export_baked": self.export_baked,
            "max_witnesses": self.max_witnesses,
            "occlusion_radius": self.occlusion_radius,
            "shading_gate": self.shading_gate,
            "depth_eps_abs": self.depth_eps_abs,
            "depth_eps_rel": self.depth_eps_rel,
            "no_data_fade": self.no_data_fade,
            "max_iterations": self.max_iterations,
            "step_alpha": self.step_alpha,
            "subpasses": self.subpasses,
            "start_jitter": self.start_jitter,
            "candidates": self.candidates_edit.text(),
            "blur_radius": self.blur_radius,
            "min_step": self.min_step,
            "grad_tol": self.grad_tol,
            "loss_tol": self.loss_tol,
            "rms_decay": self.rms_decay,
            "rms_eps": self.rms_eps,
            "use_long_bb": self.use_long_bb,
            "tile_pixels": self.tile_pixels,
            "discovery_only": self.discovery_only,
        }

    def apply_settings(self, data):
        if not data:
            return

        def number(key):
            try:
                return float(data[key])
            except (KeyError, TypeError, ValueError):
                return None

        def select(combo, key):
            if key in data:
                index = combo.findData(data[key])
                if index >= 0:
                    combo.setCurrentIndex(index)

        select(self.size_cmb, "size")
        select(self.texture_size_cmb, "texture_size")
        select(self.witness_cmb, "max_witnesses")
        select(self.tile_cmb, "tile_pixels")
        select(self.camera_cmb, "camera_key")

        has_uv = bool(self._model_has_uv[self.model_index])
        rebuild = bool(data.get("rebuild_uv", not has_uv)) or not has_uv
        self.rebuild_uv_radio.setChecked(rebuild)
        self.keep_uv_radio.setChecked(not rebuild)
        mapping = data.get("mapping_mode")
        if isinstance(mapping, int) and 0 <= mapping < len(MAPPING_MODES):
            self.mapping_cmb.setCurrentIndex(mapping)

        for widget, key in (
            (self.page_spin, "page_count"),
            (self.occlusion_radius_spin, "occlusion_radius"),
            (self.no_data_fade_spin, "no_data_fade"),
            (self.max_iters_spin, "max_iterations"),
            (self.subpasses_spin, "subpasses"),
        ):
            value = number(key)
            if value is not None:
                widget.setValue(int(value))
        for widget, key in (
            (self.pixel_size_spin, "pixel_size"),
            (self.shading_gate_spin, "shading_gate"),
            (self.depth_eps_abs_spin, "depth_eps_abs"),
            (self.depth_eps_rel_spin, "depth_eps_rel"),
            (self.step_alpha_spin, "step_alpha"),
            (self.jitter_spin, "start_jitter"),
            (self.blur_radius_spin, "blur_radius"),
            (self.min_step_spin, "min_step"),
            (self.grad_tol_spin, "grad_tol"),
            (self.loss_tol_spin, "loss_tol"),
            (self.rms_decay_spin, "rms_decay"),
            (self.rms_eps_spin, "rms_eps"),
        ):
            value = number(key)
            if value is not None:
                widget.setValue(value)
        if isinstance(data.get("candidates"), str):
            self.candidates_edit.setText(data["candidates"])
        for key, widget in (
            ("export_baked", self.export_baked_check),
            ("use_long_bb", self.long_bb_check),
            ("discovery_only", self.discovery_only_check),
        ):
            if key in data:
                widget.setChecked(bool(data[key]))
        self._sync_enabled()

    def reset_defaults(self):
        self.apply_settings(self._defaults)
        self.apply_light(self._default_light, (0.0, 0.0, 0.0))
        self.ambient_spin.setValue(0.0)

    @property
    def model_index(self):
        return self.model_cmb.currentIndex()

    @property
    def device_index(self):
        return self.device_cmb.currentIndex()

    @property
    def size(self):
        return int(self.size_cmb.currentData())

    @property
    def rebuild_uv(self):
        return self.rebuild_uv_radio.isChecked()

    @property
    def mapping_mode(self):
        return MAPPING_MODES[self.mapping_cmb.currentIndex()][1]

    @property
    def page_count(self):
        return self.page_spin.value()

    @property
    def texture_size(self):
        return int(self.texture_size_cmb.currentData())

    @property
    def pixel_size(self):
        return self.pixel_size_spin.value()

    @property
    def camera_key(self):
        if self.camera_cmb.count() == 0:
            return None
        return self.camera_cmb.currentData()

    @property
    def export_baked(self):
        return self.export_baked_check.isChecked()

    @property
    def light_multiplier(self):
        return self.light_spin.value()

    @property
    def max_witnesses(self):
        return int(self.witness_cmb.currentData())

    @property
    def occlusion_radius(self):
        return self.occlusion_radius_spin.value()

    @property
    def shading_gate(self):
        return self.shading_gate_spin.value()

    @property
    def depth_eps_abs(self):
        return self.depth_eps_abs_spin.value()

    @property
    def depth_eps_rel(self):
        return self.depth_eps_rel_spin.value()

    @property
    def no_data_fade(self):
        return self.no_data_fade_spin.value()

    @property
    def blur_radius(self):
        return self.blur_radius_spin.value()

    @property
    def discovery_only(self):
        return self.discovery_only_check.isChecked()

    @property
    def ambient(self):
        return self.ambient_spin.value()

    @property
    def subpasses(self):
        return self.subpasses_spin.value()

    @property
    def start_jitter(self):
        return self.jitter_spin.value()

    @property
    def max_iterations(self):
        return self.max_iters_spin.value()

    @property
    def step_alpha(self):
        return self.step_alpha_spin.value()

    @property
    def min_step(self):
        return self.min_step_spin.value()

    @property
    def grad_tol(self):
        return self.grad_tol_spin.value()

    @property
    def loss_tol(self):
        return self.loss_tol_spin.value()

    @property
    def rms_decay(self):
        return self.rms_decay_spin.value()

    @property
    def rms_eps(self):
        return self.rms_eps_spin.value()

    @property
    def use_long_bb(self):
        return self.long_bb_check.isChecked()

    @property
    def roughness_candidates(self):
        values = [
            value
            for value in self.candidates_edit.text().replace(";", ",").split(",")
            if value.strip()
        ]
        return [float(value) for value in values]

    @property
    def light_offset(self):
        return (
            self.light_x_spin.value(),
            self.light_y_spin.value(),
            self.light_z_spin.value(),
        )

    @property
    def tile_pixels(self):
        return int(self.tile_cmb.currentData())


def _default_bake_size(model):
    texture = model.getActiveTexture()
    if texture is None:
        return DEFAULT_BAKE_SIZE
    return min(int(texture.image().width), MAX_BAKE_SIZE)


def _model_label(model):
    name = model.label or "3D Model"
    return f"{name} ({len(model.faces):,} faces)"


def _current_model_index(chunk, models):
    active = chunk.model
    if active is not None:
        for i, model in enumerate(models):
            if model.key == active.key:
                return i
    return 0


def _camera_labels(chunk):
    return [
        ((camera.label or f"Camera {camera.key}"), camera.key)
        for camera in chunk.cameras
    ]


def _apply_uv_settings(document, chunk, model, dialog):
    if not dialog.rebuild_uv:
        return len(model.tex_vertices) > 0
    print("Rebuilding UV mapping...")
    chunk.model = None
    chunk.model = model
    mode = dialog.mapping_mode
    parameters = {"mapping_mode": mode, "page_count": dialog.page_count}
    if mode == Metashape.MappingMode.GenericMapping:
        parameters["texture_size"] = dialog.texture_size
    elif mode == Metashape.MappingMode.CameraMapping:
        if dialog.camera_key is not None:
            parameters["camera"] = dialog.camera_key
    else:
        parameters["pixel_size"] = dialog.pixel_size
    chunk.buildUV(**parameters)
    if len(model.tex_vertices) == 0:
        return False
    _save_document(document, "UV mapping saved to the project")
    return True


LIGHT_META_KEYS = (
    "LightMultiplier",
    "LightOffsetX",
    "LightOffsetY",
    "LightOffsetZ",
    "Ambient",
)


def _light_key(chunk, name):
    return f"PBR/{chunk.key}/{name}"


def _load_light_settings(document, chunk):
    settings = {"multiplier": 10.0, "offset": (0.0, 0.0, 0.0), "ambient": 0.0}
    meta = document.meta
    keys = [_light_key(chunk, name) for name in LIGHT_META_KEYS]
    try:
        if keys[0] in meta:
            settings["multiplier"] = float(meta[keys[0]])
        if keys[1] in meta:
            settings["offset"] = (
                float(meta[keys[1]]),
                float(meta[keys[2]]),
                float(meta[keys[3]]),
            )
        if keys[4] in meta:
            settings["ambient"] = float(meta[keys[4]])
    except (KeyError, ValueError):
        pass
    return settings


def _save_document(document, success_message):
    try:
        document.save(document.path if document.path else None)
        print(success_message)
    except Exception as error:
        print(f"Could not save the project ({error}); settings are kept in memory")


def _store_light_settings(document, chunk, multiplier, offset, ambient):
    meta = document.meta
    meta[_light_key(chunk, LIGHT_META_KEYS[0])] = f"{multiplier:.6f}"
    meta[_light_key(chunk, LIGHT_META_KEYS[1])] = f"{offset[0]:.6f}"
    meta[_light_key(chunk, LIGHT_META_KEYS[2])] = f"{offset[1]:.6f}"
    meta[_light_key(chunk, LIGHT_META_KEYS[3])] = f"{offset[2]:.6f}"
    meta[_light_key(chunk, LIGHT_META_KEYS[4])] = f"{ambient:.6f}"
    _save_document(document, "Light settings saved to the project")


BUILD_SETTINGS_KEY = "BuildSettings"


def _load_build_settings(document, chunk):
    meta = document.meta
    key = _light_key(chunk, BUILD_SETTINGS_KEY)
    if key not in meta:
        return None
    try:
        return json.loads(meta[key])
    except (TypeError, ValueError):
        return None


def _store_build_settings(document, chunk, settings):
    try:
        payload = json.dumps(settings)
    except (TypeError, ValueError) as error:
        print(f"Could not serialize build settings ({error})")
        return
    document.meta[_light_key(chunk, BUILD_SETTINGS_KEY)] = payload
    _save_document(document, "Build settings saved to the project")


def _bake(parent):
    chunk = Metashape.app.document.chunk
    if chunk is None or not chunk.models:
        Metashape.app.messageBox("The active chunk has no model to bake.")
        return
    models = [model for model in chunk.models if len(model.faces)]
    if not models:
        Metashape.app.messageBox("The chunk has no model to bake.")
        return

    devices = _opencl_devices()
    if not devices:
        Metashape.app.messageBox(
            "No OpenCL device found. Install a GPU driver with OpenCL support."
        )
        return
    device_names = [label for _, label in devices]

    document = Metashape.app.document
    dialog = BakeDialog(
        parent,
        [_model_label(m) for m in models],
        [len(m.tex_vertices) > 0 for m in models],
        device_names,
        _camera_labels(chunk),
        _default_bake_size(models[0]),
        default_model_index=_current_model_index(chunk, models),
        light_settings=_load_light_settings(document, chunk),
        settings=_load_build_settings(document, chunk),
    )

    def run_fit_light():
        model = models[dialog.model_index]
        if not _apply_uv_settings(document, chunk, model, dialog):
            Metashape.app.messageBox(
                "The selected model has no UV map. Enable UV rebuilding before fitting the light."
            )
            return
        dialog.mark_has_uv()
        cameras = [
            camera
            for camera in chunk.cameras
            if camera.transform is not None and camera.photo is not None
        ]
        if not cameras:
            Metashape.app.messageBox(
                "No aligned cameras in the chunk; cannot fit the light."
            )
            return
        device = devices[dialog.device_index][0]
        try:
            with _progress_session(dialog) as progress:
                fitter = _LightFitter(
                    model,
                    cameras,
                    device,
                    dialog,
                    progress.report,
                    cancel=progress.token,
                )
                multiplier, offset, error = fitter.run()
                dialog.apply_light(multiplier, offset)
                print(
                    f"Fit light: multiplier={multiplier:.4f} offset=({offset[0]:.4f},{offset[1]:.4f},{offset[2]:.4f}) "
                    f"error={error:.4f}"
                )
                _store_light_settings(
                    document, chunk, multiplier, offset, dialog.ambient
                )
        except _Cancelled:
            print("Light fitting cancelled.")
        except Exception:
            traceback.print_exc()
            Metashape.app.messageBox(
                "Light fitting failed:\n\n" + traceback.format_exc()
            )

    dialog.fit_light_button.clicked.connect(run_fit_light)

    if not dialog.exec_():
        return
    _store_light_settings(
        document, chunk, dialog.light_multiplier, dialog.light_offset, dialog.ambient
    )
    _store_build_settings(document, chunk, dialog.collect_settings())

    model = models[dialog.model_index]
    if not _apply_uv_settings(document, chunk, model, dialog):
        Metashape.app.messageBox(
            "The selected model has no UV map. Enable UV rebuilding."
        )
        return

    size = dialog.size
    out_dir = Metashape.app.getExistingDirectory("Select output folder")
    if not out_dir:
        print("No output folder selected, aborting.")
        return

    with _progress_session(parent) as progress:
        device = devices[dialog.device_index][0]
        cameras = [
            camera
            for camera in chunk.cameras
            if camera.transform is not None and camera.photo is not None
        ]
        light_multiplier = dialog.light_multiplier
        light_offset = dialog.light_offset

        print(
            f"Baking geometry maps (positions, normals, tangents) at {size}x{size}..."
        )
        positions, normals, tangents = _bake_geometry(
            model, size, progress.report, cancel=progress.token
        )
        print(f"Baked {int((~np.isnan(positions[..., 0])).sum())} valid texels")

        if dialog.export_baked:
            _save_baked_maps(out_dir, positions, normals, tangents)

        if not cameras:
            print("No aligned cameras in the chunk; cannot gather witnesses.")
            return
        print(f"Gathering witnesses from {len(cameras)} cameras at {size}x{size}...")
        started = time.time()
        coverage, image_lights, pool, counts = _gather(
            model,
            positions,
            normals,
            cameras,
            size,
            device,
            light_multiplier,
            light_offset,
            dialog.max_witnesses,
            progress.report,
            dialog.occlusion_radius,
            dialog.shading_gate,
            dialog.depth_eps_abs,
            dialog.depth_eps_rel,
            float(dialog.no_data_fade),
            cancel=progress.token,
        )
        print(f"Witness gathering finished in {time.time() - started:.1f}s")
        print(f"Recorded {len(image_lights)} camera-light pairs")
        _save_coverage(out_dir, coverage)

        print("Running discovery pass...")
        albedo, roughness, metallic, normal_map, errors = _run_build(
            positions,
            normals,
            tangents,
            image_lights,
            pool,
            counts,
            size,
            device,
            dialog,
            progress.report,
            cancel=progress.token,
        )
        _save_discovery_maps(
            out_dir, size, albedo, roughness, metallic, normal_map, errors
        )


def build_pbr_textures():
    try:
        parent = QtWidgets.QApplication.instance().activeWindow()
        _bake(parent)
    except _Cancelled:
        print("Build cancelled.")
    except Exception:
        traceback.print_exc()
        Metashape.app.messageBox("PBR Builder failed:\n\n" + traceback.format_exc())


label = "Scripts/Build PBR textures"
Metashape.app.addMenuItem(label, build_pbr_textures)
print(f"To execute this script press {label}")
