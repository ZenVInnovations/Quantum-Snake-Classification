"""
features.py — turn a photo into a numeric feature vector.

Quantum circuits (and the classical baselines) work on numbers, not raw pixels,
so every image must first be converted to a feature vector. Two extractors:

  extract_features_cnn(path)    <- RECOMMENDED for real use.
      A pretrained MobileNetV2 (ImageNet) used as a frozen feature extractor,
      giving a 1280-dim embedding per photo. Needs torch + torchvision; the
      weights download once, automatically, the first time you run it (so you
      need internet the first time).

  extract_features_simple(path) <- for a quick, no-download smoke test.
      Colour histograms + basic texture/colour statistics using only Pillow +
      NumPy. Works offline anywhere, but is MUCH weaker than the CNN — use it
      only to check the pipeline end-to-end, not for real accuracy.

Both return a 1-D float64 numpy array. `extract_folder` walks a dataset folder.
"""

import os
import glob
import numpy as np
from PIL import Image

_CNN = None   # cached model
_MEAN = None  # cached ImageNet mean tensor
_STD = None   # cached ImageNet std tensor


def _load_rgb(path, size):
    return Image.open(path).convert("RGB").resize(size)


def extract_features_simple(path):
    """Colour-histogram + texture features. No downloads. Weak but portable."""
    arr = np.asarray(_load_rgb(path, (128, 128)), dtype=np.float32) / 255.0
    parts = []
    for c in range(3):                                   # per-channel histogram
        hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)
        parts.append(hist)
    gray = arr.mean(axis=2)                              # simple texture
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    parts.append(np.array([gx.mean(), gx.std(), gy.mean(), gy.std(),
                           gray.mean(), gray.std()]))
    flat = arr.reshape(-1, 3)                            # colour moments
    parts.append(flat.mean(axis=0))
    parts.append(flat.std(axis=0))
    return np.concatenate([p.ravel() for p in parts]).astype(np.float64)


def extract_features_cnn(path):
    """Pretrained MobileNetV2 embedding (1280-dim). NumPy-2.x safe.

    Uses explicit float32 pre-processing instead of torchvision's built-in
    transform, because that transform converts images to uint8 internally and
    newer NumPy (2.x) then raises "Could not infer dtype of numpy.uint8".
    """
    global _CNN, _MEAN, _STD
    try:
        import torch
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    except ImportError as e:
        raise ImportError(
            "The CNN extractor needs PyTorch. Install it with:\n"
            "    pip install torch torchvision\n"
            "or use --extractor simple for an offline smoke test."
        ) from e

    if _CNN is None:
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        model.classifier = torch.nn.Identity()           # drop the 1000-class head
        model.eval()
        _CNN = model
        _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    img = _load_rgb(path, (224, 224))
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.float32) / 255.0)  # H,W,C
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # 1,C,H,W float32
    t = (t - _MEAN) / _STD                                    # ImageNet normalise
    with torch.no_grad():
        emb = _CNN(t).squeeze(0).numpy()
    return emb.astype(np.float64)


def get_extractor(name):
    return {"cnn": extract_features_cnn, "simple": extract_features_simple}[name]


def extract_folder(root, extractor):
    """Read root/venomous/* and root/non_venomous/*  ->  X, y, paths.

    Labels:  non_venomous = 0,  venomous = 1.
    """
    # accept both "venomous"/"non_venomous" and "Venomous"/"Non Venomous"
    class_aliases = {
        1: ["venomous", "Venomous", "VENOMOUS"],
        0: ["non_venomous", "Non Venomous", "non venomous", "NonVenomous", "Non_Venomous"],
    }
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    X, y, paths = [], [], []
    # walk recursively so train/ and test/ subfolders are both included
    for label, names in class_aliases.items():
        dirs = []
        for dirpath, dirnames, _ in os.walk(root):
            for dn in dirnames:
                if dn in names:
                    dirs.append(os.path.join(dirpath, dn))
        for d in dirs:
        files = []
        for e in exts:
            files += glob.glob(os.path.join(d, e))
            files += glob.glob(os.path.join(d, e.upper()))
        for f in sorted(set(files)):
            try:
                X.append(extractor(f)); y.append(label); paths.append(f)
            except Exception as ex:                       # skip unreadable files
                print(f"  ! skipped {f}: {ex}")
    if not X:
        raise RuntimeError(
            f"No images found under {root}. Expected subfolders "
            f"'venomous/' and 'non_venomous/' containing image files."
        )
    return np.array(X), np.array(y), paths
