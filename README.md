# Python scripts for Metashape Pro

These scripts are supposed to be used in Metashape Pro. They can be executed on Metashape launch automatically ([see article](https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start)).

# Contributions are welcome!

If you have script with general functionality that can be useful for other users - feel free to publish your script in this repository in directory [contrib](https://github.com/agisoft-llc/metashape-scripts/tree/master/src/contrib) via pull request.

# Compatibility with Metashape versions and PhotoScan

Each script checks that it is used with the proper Metashape version. Main branch of repository contains scripts compatible with last released version of Metashape.

If you need scripts for older releases of Metashape or PhotoScan - you can use scripts from the proper branch.

---

# Installazione su macOS — Procedura completa

> Testato su: **macOS** · **Metashape Professional 2.3.1** · **Python 3.12.12** (interno a Metashape)
> Architettura: Intel x86_64

## Prerequisiti

- Metashape Professional installato in `/Applications/MetashapePro.app`
- Xcode Command Line Tools installati (`xcode-select --install`)
- Git installato

## 1. Clonare il repository

```zsh
git clone --depth 1 https://github.com/agisoft-llc/metashape-scripts.git ~/metashape-scripts
```

## 2. Installare gli script nel percorso di auto-avvio di Metashape

Metashape carica automaticamente tutti gli `.py` presenti in questa cartella:

```zsh
SCRIPTS_DIR="$HOME/Library/Application Support/Agisoft/Metashape Pro/scripts"
mkdir -p "$SCRIPTS_DIR"

# Script principali (aggiungono voci nel menu Custom menu / Tools)
cp ~/metashape-scripts/src/*.py "$SCRIPTS_DIR/"

# Script contrib (community)
mkdir -p "$SCRIPTS_DIR/contrib"
cp ~/metashape-scripts/src/contrib/*.py "$SCRIPTS_DIR/contrib/"
```

## 3. Correzione SDK Xcode (necessaria su macOS con solo Command Line Tools)

Il Python interno di Metashape è stato compilato con riferimento a `MacOSX11.1.sdk`.
Se Xcode.app non è installato (solo Command Line Tools), la compilazione di estensioni C++ fallisce.
Creare un symlink una-tantum che risolve il problema:

```zsh
sudo mkdir -p "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs"
sudo ln -s \
  "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" \
  "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.1.sdk"
```

> Questo symlink è innocuo e persistente. Non richiede Xcode.app installato.

## 4. Installare le dipendenze Python

Le dipendenze vengono installate nell'ambiente isolato di Metashape (`user-packages-py312`),
senza toccare il Python di sistema.

```zsh
PYBIN="/Applications/MetashapePro.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
USERBASE="$HOME/Library/Application Support/Agisoft/Metashape Pro/user-packages-py312"
SDK="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.1.sdk"

# Dipendenze generali (masking, allineamento modelli, depth maps)
PYTHONUSERBASE="$USERBASE" "$PYBIN" -m pip install --no-warn-script-location --user \
  -r ~/metashape-scripts/mac_requirements.txt

# Dipendenze detect_objects (torch CPU + deepforest)
SDKROOT="$SDK" MACOSX_DEPLOYMENT_TARGET=10.15 \
PYTHONUSERBASE="$USERBASE" "$PYBIN" -m pip install --no-warn-script-location --user \
  --index-url https://pypi.org/simple \
  "faster-coco-eval>=1.6.8" \
  torch==2.2.2 \
  torchvision==0.17.2 \
  "deepforest==2.0.0" \
  "pytorch-lightning==2.6.1" \
  "albumentations==2.0.8"
```

Il file `mac_requirements.txt` (incluso in questo repo) contiene i pacchetti pinnati
per le librerie generali (numpy, onnxruntime, opencv, Pillow, open3d, scipy, rasterio, ecc.).

## 5. Modifica detect_objects.py per macOS (rimuove vincolo CUDA)

Lo script originale richiede `torch==2.7.0+cu128` (CUDA, solo Linux/Windows con GPU NVIDIA).
La copia installata in Metashape va modificata per usare la versione CPU.

Nel file `"$SCRIPTS_DIR/detect_objects.py"` sostituire il blocco `requirements_txt` (righe 80-91):

```python
# SOSTITUZIONE (CPU, compatibile macOS Intel):
requirements_txt = """--index-url https://pypi.org/simple
torch==2.2.2
torchvision==0.17.2

deepforest==2.0.0
pytorch-lightning==2.6.1
albumentations==2.0.8

rasterio==1.4.3"""
```

> `detect_objects.py` funzionerà in modalità **CPU only** (più lento, ma funzionale).
> Lo script gestisce già il fallback automatico a CPU (stampa `"Using CPU (will be very slow)..."`).

## 6. Verifica installazione

### Test da terminale (senza GUI)

```zsh
PYTHONUSERBASE="$HOME/Library/Application Support/Agisoft/Metashape Pro/user-packages-py312" \
/Applications/MetashapePro.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
~/metashape-scripts/test_libraries.py
```

Output atteso (13/13 test superati):

```
Python: 3.12.12  |  Piattaforma: darwin

TEST LIBRERIE GENERALI
  ✓ numpy                          v1.26.4        (0.5s)
  ✓ Pillow                         v10.3.0        (0.9s)
  ✓ opencv-python (cv2)            v4.11.0        (0.4s)
  ✓ scipy (ConvexHull)             v1.12.0        (0.9s)
  ✓ onnxruntime                    v1.18.1        (3.1s)
  ✓ rasterio (GDAL bundled)        v1.4.3         (0.9s)
  ✓ open3d (PointCloud)            v0.19.0       (27.2s)

TEST STACK DETECT_OBJECTS (torch / deepforest)
  ✓ torch                          v2.2.2 [CPU]   (4.4s)
  ✓ torchvision (transforms)       v0.17.2        (4.0s)
  ✓ albumentations                 v2.0.8         (2.2s)
  ✓ pytorch-lightning              v2.6.1         (3.5s)
  ✓ deepforest (import)            v2.0.0         (0.0s)
  ✓ faster-coco-eval               v1.7.2         (1.6s)

RISULTATO: 13/13 test superati
✓ Tutte le librerie sono installate e funzionanti.
```

### Test da interfaccia Metashape (con GUI)

1. Aprire Metashape Pro
2. Nel menu: **Custom menu → Test integrazione librerie**
3. Si apre una dialog con gli stessi risultati, inclusa la verifica dell'API `Metashape.*`

---

## Note su compatibilità macOS

| Componente | Stato | Note |
|---|---|---|
| numpy, scipy, Pillow, opencv | ✅ Funzionante | Wheel precompilati |
| onnxruntime | ✅ Funzionante | Usa CoreML (Neural Engine) automaticamente |
| rasterio | ✅ Funzionante | GDAL 3.9.3 incluso nel wheel |
| open3d | ✅ Funzionante | Primo import lento (~27s), poi nella norma |
| torch | ✅ Funzionante (CPU) | Max v2.2.2 su macOS Intel; PyTorch ha abbandonato x86_64 dopo la 2.2.x |
| CUDA / GPU | ❌ Non supportato | macOS Intel non ha GPU NVIDIA; detect_objects usa CPU |
| MPS (Apple Silicon) | ❌ Non applicabile | Questo Mac è Intel x86_64 |

## Struttura file installati

```
~/metashape-scripts/
├── README.md                    ← questo file
├── mac_requirements.txt         ← dipendenze generali pinnate per macOS
├── test_libraries.py            ← test da terminale (senza GUI)
└── src/
    ├── *.py                     ← 28 script principali
    ├── contrib/                 ← 12 script community
    └── samples/                 ← workflow di esempio (non installati)

~/Library/Application Support/Agisoft/Metashape Pro/
├── scripts/
│   ├── *.py                     ← script caricati da Metashape all'avvio
│   ├── contrib/
│   └── test_integration.py      ← test GUI (Custom menu > Test integrazione)
└── user-packages-py312/         ← dipendenze Python isolate (gestite da pip)
    └── lib/python3.12/site-packages/
        ├── torch/               ← 2.2.2 CPU
        ├── deepforest/          ← 2.0.0
        ├── open3d/              ← 0.19.0
        └── ...                  ← tutte le altre librerie
```
