# Quantum vs Classical Snake Classification

Benchmark of NISQ-era quantum ML (VQC, QCNN, QSVM) against a classical SVM
baseline for venomous/non-venomous snake identification. Companion code for
the manuscript.

## Data
Download from https://www.kaggle.com/datasets/adityasharma01/snake-dataset-india
Unzip so `Snake Images/` (with train/ and test/ subfolders) sits in this directory.

## Install
    python -m venv venv && source venv/bin/activate
    pip install torch torchvision scikit-learn pennylane numpy joblib pillow

## Reproduce Table 1 (main result)
    # 1. Extract CNN features -> snake_cnn_features.npz
    python build_features.py --data "Snake Images"

    # 2. 5-fold cross-validated results (classical baseline on full data,
    #    QSVM matched at C=3.0, evaluated on the full set via analytic kernel)
    python reproduce_paper.py --features snake_cnn_features.npz --qsvm_train 0 --qsvm_test 0

Expected: classical SVM ~85%, QSVM ~79%, matched 8-PCA classical control ~80%,
majority-class baseline 58.8%.

## Other scripts
    quantum_kernel.py   - faster single-split QSVM variant (higher variance)
    cross_validate.py   - variational models (VQC, QCNN)
    qubit_scaling_cv.py - accuracy vs qubit count

Quantum circuits run on PennyLane's default.qubit simulator (noiseless).
