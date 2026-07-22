import time, numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pennylane as qml
from pennylane import numpy as pnp

SEED = 0
N_FOLDS = 5
LAYERS = 3
EPOCHS = 40
LR = 0.05

def amp_prep(X, nq):
    X = np.asarray(X, float)
    cap = 2 ** nq
    if X.shape[1] > cap:
        X = X[:, :cap]
    elif X.shape[1] < cap:
        X = np.pad(X, ((0, 0), (0, cap - X.shape[1])))
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return pnp.array(X / n, requires_grad=False)

def train_vqc(Xtr, ytr, Xte, nq):
    dev = qml.device("default.qubit", wires=nq)
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circ(x, w):
        qml.AmplitudeEmbedding(x, wires=range(nq), normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(w, wires=range(nq))
        return qml.expval(qml.PauliZ(0))
    Xb = amp_prep(Xtr, nq)
    yb = pnp.array(2 * np.asarray(ytr, float) - 1, requires_grad=False)
    shape = qml.StronglyEntanglingLayers.shape(LAYERS, nq)
    rng = np.random.default_rng(SEED)
    w = pnp.array(0.1 * rng.standard_normal(shape), requires_grad=True)
    opt = qml.AdamOptimizer(LR)
    def cost(w):
        return pnp.mean((circ(Xb, w) - yb) ** 2)
    for _ in range(EPOCHS):
        w, _ = opt.step_and_cost(cost, w)
    return (np.asarray(circ(amp_prep(Xte, nq), w)).ravel() > 0).astype(int)

def main():
    print("=" * 62)
    print("QUBIT-SCALING under 5-FOLD CROSS-VALIDATION (Issue 1)")
    print("=" * 62)
    d = np.load("snake_cnn_features.npz")
    X, y = d["X"], d["y"]
    print(f"Dataset: {X.shape[0]} images, {X.shape[1]} features")
    print("Running VQC at 8 and 12 qubits across 5 folds each...\n")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    res = {8: [], 12: []}
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for nq in (8, 12):
            pred = train_vqc(Xtr, y[tr], Xte, nq)
            acc = accuracy_score(y[te], pred)
            res[nq].append(acc)
            print(f"  Fold {fold}  VQC {nq:2d} qubits: {acc*100:5.1f}%", flush=True)
        print(f"    (fold done in {time.time()-t0:.0f}s)\n", flush=True)
    a8, a12 = np.array(res[8]) * 100, np.array(res[12]) * 100
    print("=" * 62)
    print("RESULTS (mean +/- std over 5 folds)")
    print("=" * 62)
    print(f"  VQC  8 qubits: {a8.mean():5.1f} +/- {a8.std():.1f}%")
    print(f"  VQC 12 qubits: {a12.mean():5.1f} +/- {a12.std():.1f}%")
    print(f"  Change (12 - 8): {a12.mean()-a8.mean():+.1f} points")
    t, p = stats.ttest_rel(a8, a12)
    print(f"  Paired t-test: p = {p:.4f}")
    print()
    if a12.mean() < a8.mean():
        print("  => 12 qubits is WORSE than 8 under CV (confirms single-split trend)")
    else:
        print("  => 12 qubits is NOT worse under CV (single-split trend was noise!)")
    print("  Interpret p conservatively (n=5 folds).")
    np.savez("qubit_scaling_cv.npz", q8=res[8], q12=res[12])
    print("\nFOR THE PAPER: VQC 8q = {:.1f}+/-{:.1f}%, 12q = {:.1f}+/-{:.1f}%, p={:.3f}".format(
        a8.mean(), a8.std(), a12.mean(), a12.std(), p))

if __name__ == "__main__":
    main()
