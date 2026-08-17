# Quantum vs Classical Snake Classification

Benchmark of NISQ-era quantum ML (VQC, QCNN, QSVM) against a classical
SVM baseline for venomous/non-venomous snake identification.

## Data
Download from https://www.kaggle.com/datasets/adityasharma01/snake-dataset-india
Unzip so `Snake Images/` (with train/ and test/ subfolders) sits in this directory.

## Install
    python -m venv venv && source venv/bin/activate
    pip install torch torchvision scikit-learn pennylane numpy joblib pillow

## Reproduce
    # 1. Extract CNN features (creates snake_cnn_features.npz)
    python features.py

    # 2. Classical + quantum kernel comparison
    python quantum_kernel.py --qubits 8

    # 3. Variational models
    python cross_validate.py

## Notes
Quantum circuits run on PennyLane's default.qubit simulator (noiseless).
Results reproduce Tables 1-3 of the manuscript.
