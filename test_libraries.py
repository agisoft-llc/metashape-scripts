#!/usr/bin/env python3
"""
Test di verifica per le librerie installate nell'ambiente Python di Metashape Pro.
Eseguire con:
  PYTHONUSERBASE="$HOME/Library/Application Support/Agisoft/Metashape Pro/user-packages-py312" \
  /Applications/MetashapePro.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  ~/metashape-scripts/test_libraries.py
"""

import sys
import time

PASS = "  ✓"
FAIL = "  ✗"
results = []

def test(name, fn):
    t0 = time.time()
    try:
        info = fn()
        elapsed = time.time() - t0
        print(f"{PASS} {name:<30} {info}  ({elapsed:.1f}s)")
        results.append((name, True))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{FAIL} {name:<30} ERRORE: {e}  ({elapsed:.1f}s)")
        results.append((name, False))

print(f"\nPython: {sys.version.split()[0]}  |  Piattaforma: {sys.platform}\n")
print("=" * 70)
print("TEST LIBRERIE GENERALI")
print("=" * 70)

# numpy
def t_numpy():
    import numpy as np
    a = np.array([1.0, 2.0, 3.0])
    assert a.sum() == 6.0
    return f"v{np.__version__}"
test("numpy", t_numpy)

# Pillow
def t_pillow():
    from PIL import Image
    import io, PIL
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert buf.tell() > 0
    return f"v{PIL.__version__}"
test("Pillow", t_pillow)

# opencv
def t_opencv():
    import cv2, numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (100, 100)
    return f"v{cv2.__version__}"
test("opencv-python (cv2)", t_opencv)

# scipy
def t_scipy():
    import scipy
    from scipy.spatial import ConvexHull
    import numpy as np
    pts = np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])
    hull = ConvexHull(pts)
    assert len(hull.vertices) >= 3
    return f"v{scipy.__version__}"
test("scipy (ConvexHull)", t_scipy)

# onnxruntime
def t_onnx():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    return f"v{ort.__version__}  providers={providers}"
test("onnxruntime", t_onnx)

# rasterio
def t_rasterio():
    import rasterio
    return f"v{rasterio.__version__} (GDAL {rasterio.__gdal_version__})"
test("rasterio (GDAL bundled)", t_rasterio)

# open3d
def t_open3d():
    import open3d as o3d
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    assert len(pcd.points) == 100
    return f"v{o3d.__version__}"
test("open3d (PointCloud)", t_open3d)

print()
print("=" * 70)
print("TEST STACK DETECT_OBJECTS (torch / deepforest)")
print("=" * 70)

# torch
def t_torch():
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    assert float(t.sum()) == 6.0
    cuda = torch.cuda.is_available()
    mode = "CUDA" if cuda else "CPU"
    return f"v{torch.__version__} [{mode}]"
test("torch", t_torch)

# torchvision
def t_torchvision():
    import torchvision
    import torch
    # Test transforms pipeline
    from torchvision import transforms
    from PIL import Image
    t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    img = Image.new("RGB", (64, 64))
    tensor = t(img)
    assert tensor.shape == (3, 32, 32)
    return f"v{torchvision.__version__}"
test("torchvision (transforms)", t_torchvision)

# albumentations
def t_albumentations():
    import albumentations as A
    import numpy as np
    transform = A.Compose([A.HorizontalFlip(p=1.0)])
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = transform(image=img)
    assert result["image"].shape == (64, 64, 3)
    return f"v{A.__version__}"
test("albumentations", t_albumentations)

# pytorch_lightning
def t_lightning():
    import pytorch_lightning as pl
    return f"v{pl.__version__}"
test("pytorch-lightning", t_lightning)

# deepforest
def t_deepforest():
    import deepforest
    return f"v{deepforest.__version__}"
test("deepforest (import)", t_deepforest)

# faster_coco_eval
def t_fce():
    import faster_coco_eval
    return f"v{faster_coco_eval.__version__}"
test("faster-coco-eval", t_fce)

# --- Sommario ---
print()
print("=" * 70)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"RISULTATO: {passed}/{total} test superati")
if passed == total:
    print("✓ Tutte le librerie sono installate e funzionanti.")
else:
    failed = [name for name, ok in results if not ok]
    print(f"✗ Falliti: {', '.join(failed)}")
print("=" * 70)
sys.exit(0 if passed == total else 1)
