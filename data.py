"""
data.py — multi-modal dataset for snake detection / species identification.

IMPORTANT (honesty note):
    No public multi-modal snake dataset exists — image corpora like SnakeCLEF
    exist, but a paired *acoustic + vibration* snake dataset does not (see the
    project literature review). So this module generates a STRUCTURED SYNTHETIC
    dataset that mimics the shape of the real problem: two feature "modalities"
    (image-derived + acoustic/vibration-derived), an informative-plus-noisy
    feature mix, label noise, and a nonlinear interaction so that nonlinear
    models (and, potentially, quantum models) have headroom over linear ones.

    It is scaffolding for the *method*, not a claim about real performance.
    To use real data, implement `load_real_features()` below and return the
    same (X, y, modality_slices) contract.

Primary task: BINARY  ->  venomous (1) vs non-venomous (0).
    This is the high-value, tractable target argued for throughout the project;
    fine-grained multi-class species ID is a documented extension.
"""

import numpy as np
from sklearn.datasets import make_classification

# Column layout of the fused feature vector
N_IMAGE_FEATS = 16      # e.g. embeddings from a CNN backbone on a snake photo
N_ACOUSTIC_FEATS = 8    # e.g. MFCC / spectral + vibration-band features
N_FEATURES = N_IMAGE_FEATS + N_ACOUSTIC_FEATS

MODALITY_SLICES = {
    "image": slice(0, N_IMAGE_FEATS),
    "acoustic_vibration": slice(N_IMAGE_FEATS, N_FEATURES),
}


def make_multimodal_snake_data(n_samples=600, seed=0, label_noise=0.06):
    """Return (X, y, modality_slices) for the binary venomous-vs-non task.

    X: (n_samples, N_FEATURES) float array — image feats then acoustic feats.
    y: (n_samples,) int array in {0, 1}.
    """
    rng = np.random.default_rng(seed)

    # Core structured signal: informative + redundant + pure-noise features,
    # two clusters per class (so the boundary is nonlinear), mild class overlap.
    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=8,
        n_redundant=6,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.15,
        flip_y=label_noise,       # realistic label noise
        random_state=seed,
    )

    # Make the two modalities behave differently, as they would in reality:
    #  - image block: relatively clean, strong signal
    #  - acoustic/vibration block: weaker, noisier (snakes are mostly quiet)
    img = X[:, MODALITY_SLICES["image"]]
    aco = X[:, MODALITY_SLICES["acoustic_vibration"]]
    aco = aco + rng.normal(0, 0.9, size=aco.shape)          # extra sensor noise
    aco *= 0.7                                              # weaker amplitude

    # Inject an explicit nonlinear (XOR-like) interaction between one image and
    # one acoustic feature, so linear models leave accuracy on the table and
    # nonlinear/quantum models can, in principle, recover it.
    interaction = np.sign(img[:, 0]) * np.sign(aco[:, 0])
    img[:, 1] += 1.1 * interaction

    X = np.hstack([img, aco]).astype(np.float64)

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm].astype(int), MODALITY_SLICES


def load_real_features():
    """Plug-in point for REAL data — not implemented.

    Expected contract (mirror make_multimodal_snake_data):
        return X (n, d), y (n,) in {0,1}, modality_slices dict.

    Suggested construction:
      * image features  : penultimate-layer embeddings from a CNN / ViT
                          fine-tuned on a snake corpus (e.g. SnakeCLEF),
                          run over each photo.
      * acoustic/vibration features : MFCCs + log-mel + vibration-band
                          (approx. 1-50 Hz) statistics from your sensor node,
                          IF/when you have a labelled recording set.
      * y : 1 = venomous, 0 = non-venomous (or look-alike).
    """
    raise NotImplementedError(
        "Provide real features here — see docstring for the expected format."
    )
