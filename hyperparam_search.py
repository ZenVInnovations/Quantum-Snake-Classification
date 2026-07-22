"""
hyperparam_search.py — VQC hyperparameter sensitivity (reviewer Issue 3).

Shows that no reasonable hyperparameter choice closes the 20.5-point gap to the
classical baseline. Sweeps learning rate, epochs, and layer count for the VQC on
a single stratified split (exploratory, as stated in the paper), reporting test
accuracy for each configuration.

Run:  python hyperparam_search.py
Takes ~20-40 min depending on grid size.
"""
import time, itertools, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pennylane as qml
from pennylane import numpy as pnp

SEED = 0
NQ = 8

# the grid (kept modest so it runs in reasonable time)
LR_GRID = [0.01, 0.05, 0.1]
EPOCH_GRID = [40, 80, 150]
LAYER_GRID = [2, 3, 5]


def amp_prep(X, nq):
    X = np.asarray(X, float)
    cap = 2 ** nq
    if X.shape[1] > cap:
        X = X[:, :cap]
    elif X.shape[1] < cap:
        X = np.pad(X, ((0, 0), (0, cap - X.shape[1])))
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return pnp.array(X / n, requires_grad=False)


def train_vqc(Xtr, ytr, Xte, nq, layers, epochs, lr):
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circ(x, w):
        qml.AmplitudeEmbedding(x, wires=range(nq), normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(w, wires=range(nq))
        return qml.expval(qml.PauliZ(0))

    Xb = amp_prep(Xtr, nq)
    yb = pnp.array(2 * np.asarray(ytr, float) - 1, requires_grad=False)
    shape = qml.StronglyEntanglingLayers.shape(layers, nq)
    rng = np.random.default_rng(SEED)
    w = pnp.array(0.1 * rng.standard_normal(shape), requires_grad=True)
    opt = qml.AdamOptimizer(lr)

    def cost(w):
        return pnp.mean((circ(Xb, w) - yb) ** 2)

    for _ in range(epochs):
        w, _ = opt.step_and_cost(cost, w)
    return (np.asarray(circ(amp_prep(Xte, nq), w)).ravel() > 0).astype(int)


def main():
    print("=" * 62)
    print("VQC HYPERPARAMETER SENSITIVITY (Issue 3)")
    print("=" * 62)
    d = np.load("snake_cnn_features.npz")
    X, y = d["X"], d["y"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    print(f"Single stratified split: {len(Xtr)} train / {len(Xte)} test")
    print(f"Grid: {len(LR_GRID)}x{len(EPOCH_GRID)}x{len(LAYER_GRID)} = "
          f"{len(LR_GRID)*len(EPOCH_GRID)*len(LAYER_GRID)} configurations")
    print("Classical SVM baseline on this split for reference:")

    from sklearn.svm import SVC
    svm = SVC(kernel="rbf", C=3.0, gamma="scale", random_state=SEED).fit(Xtr, ytr)
    acc_c = accuracy_score(yte, svm.predict(Xte))
    print(f"  Classical SVM (full feats): {acc_c*100:.1f}%\n")

    results = []
    best = (0, None)
    t_start = time.time()
    for lr, ep, ly in itertools.product(LR_GRID, EPOCH_GRID, LAYER_GRID):
        t0 = time.time()
        pred = train_vqc(Xtr, ytr, Xte, NQ, ly, ep, lr)
        acc = accuracy_score(yte, pred)
        results.append((lr, ep, ly, acc))
        if acc > best[0]:
            best = (acc, (lr, ep, ly))
        print(f"  lr={lr:<4} epochs={ep:<3} layers={ly}: "
              f"{acc*100:5.1f}%   ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    accs = np.array([r[3] for r in results]) * 100
    print(f"  VQC best config:  {best[0]*100:.1f}%  (lr={best[1][0]}, "
          f"epochs={best[1][1]}, layers={best[1][2]})")
    print(f"  VQC worst config: {accs.min():.1f}%")
    print(f"  VQC mean:         {accs.mean():.1f}% +/- {accs.std():.1f}")
    print(f"  Classical SVM:    {acc_c*100:.1f}%")
    print(f"  Gap even at BEST VQC config: {acc_c*100 - best[0]*100:.1f} points")
    print()
    print("  => If the best config still trails classical by a wide margin,")
    print("     under-training / hyperparameters do NOT explain the gap.")
    np.savez("hyperparam_results.npz",
             results=np.array([(r[0], r[1], r[2], r[3]) for r in results]),
             classical=acc_c)
    print(f"\nFOR THE PAPER: best VQC = {best[0]*100:.1f}%, classical = {acc_c*100:.1f}%, "
          f"gap = {acc_c*100-best[0]*100:.1f} pts across {len(results)} configs")


if __name__ == "__main__":
    main()
