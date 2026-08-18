"""build_features.py - create snake_cnn_features.npz from the image folders.

    python build_features.py --data "Snake Images"

Produces snake_cnn_features.npz (keys: X, y) used by quantum_kernel.py,
cross_validate.py, and the other experiment scripts.
"""
import argparse, numpy as np
from features import extract_folder, get_extractor

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="Snake Images",
                help="dataset root containing venomous/non_venomous folders")
ap.add_argument("--extractor", default="cnn", choices=["cnn", "simple"])
ap.add_argument("--out", default="snake_cnn_features.npz")
a = ap.parse_args()

print(f"Extracting '{a.extractor}' features from {a.data} ...")
X, y, paths = extract_folder(a.data, get_extractor(a.extractor))
np.savez(a.out, X=X, y=y)
print(f"  {X.shape[0]} images, {X.shape[1]} features each")
print(f"  venomous={int((y==1).sum())}, non-venomous={int((y==0).sum())}")
print(f"Saved -> {a.out}")
