import argparse, time, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pennylane as qml

SEED = 0

def build_kernel_fn(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
        return qml.probs(wires=range(n_qubits))
    def quantum_kernel(x1, x2):
        return float(kernel_circuit(x1, x2)[0])
    return quantum_kernel

def build_kernel_matrix(X1, X2, kernel_fn, label=""):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2)); done = 0; t0 = time.time()
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j]); done += 1
            if done % 500 == 0:
                eta = (time.time()-t0)/done*(n1*n2-done)
                print(f"  {label} {done}/{n1*n2} ({done*100//(n1*n2)}%)  ETA {eta/60:.1f} min ...")
    return K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="snake_cnn_features.npz")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--out", default="snake_model_qk.joblib")
    args = ap.parse_args()
    N = args.qubits
    print("=" * 60)
    print("Quantum Kernel Method for Snake Classification")
    print("=" * 60)
    print(f"\n[1/5] Loading CNN features from {args.features} ...")
    d = np.load(args.features)
    X, y = d["X"], d["y"]
    print(f"      {X.shape[0]} photos | {X.shape[1]} features each")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    X_use = X[:args.n_train+200]
    y_use = y[:args.n_train+200]
    print(f"      Using {args.n_train} train + 200 test samples")
    X_tr,X_te,y_tr,y_te = train_test_split(
        X_use, y_use, test_size=200, stratify=y_use, random_state=SEED)
    print(f"\n[2/5] Preprocessing ...")
    scaler = StandardScaler().fit(X_tr)
    Xtr_s = scaler.transform(X_tr)
    Xte_s = scaler.transform(X_te)
    pca = PCA(n_components=N, random_state=SEED).fit(Xtr_s)
    ev = pca.explained_variance_ratio_.sum()
    Xtr_p = pca.transform(Xtr_s)
    Xte_p = pca.transform(Xte_s)
    qsc = MinMaxScaler((0, np.pi)).fit(Xtr_p)
    Xtr_q = qsc.transform(Xtr_p)
    Xte_q = qsc.transform(Xte_p)
    print(f"      PCA-{N} retains {ev*100:.1f}% of variance")
    print(f"\n[3/5] Building {N}-qubit quantum kernel matrices ...")
    print(f"      Train: {len(Xtr_q)}x{len(Xtr_q)} = {len(Xtr_q)**2:,} evals")
    print(f"      Test:  {len(Xte_q)}x{len(Xtr_q)} = {len(Xte_q)*len(Xtr_q):,} evals")
    kernel_fn = build_kernel_fn(N)
    t0 = time.time()
    K_train = build_kernel_matrix(Xtr_q, Xtr_q, kernel_fn, "train kernel")
    print(f"      Train kernel done in {(time.time()-t0)/60:.1f} min")
    t1 = time.time()
    K_test = build_kernel_matrix(Xte_q, Xtr_q, kernel_fn, "test kernel")
    print(f"      Test kernel done in {(time.time()-t1)/60:.1f} min")
    print("\n[4/5] Classical SVM baseline ...")
    svm_c = SVC(kernel="rbf", C=3.0, gamma="scale",
                random_state=SEED).fit(Xtr_s, y_tr)
    acc_c = accuracy_score(y_te, svm_c.predict(Xte_s))
    f1_c = f1_score(y_te, svm_c.predict(Xte_s), zero_division=0)
    print("[5/5] Training SVM on quantum kernel matrix ...")
    svm_qk = SVC(kernel="precomputed", C=1.0,
                 random_state=SEED).fit(K_train, y_tr)
    preds = svm_qk.predict(K_test)
    acc_qk = accuracy_score(y_te, preds)
    f1_qk = f1_score(y_te, preds, zero_division=0)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Quantum Kernel SVM ({N}q)     acc={acc_qk*100:5.1f}%   F1={f1_qk*100:5.1f}%")
    print(f"  Classical SVM (full feats)  acc={acc_c*100:5.1f}%   F1={f1_c*100:5.1f}%")
    print("=" * 60)
    gap = (acc_c - acc_qk) * 100
    if abs(gap) < 3:
        print("\nVerdict: Quantum kernel is COMPETITIVE with classical SVM")
    elif gap > 0:
        print(f"\nVerdict: Classical SVM leads by {gap:.1f} points")
    else:
        print(f"\nVerdict: Quantum kernel edges ahead by {abs(gap):.1f} points")
    joblib.dump({"svm_qk": svm_qk, "svm_classical": svm_c,
        "scaler": scaler, "pca": pca, "qscaler": qsc,
        "n_qubits": N, "acc_qk": acc_qk, "f1_qk": f1_qk,
        "acc_classical": acc_c, "f1_classical": f1_c}, args.out)
    print(f"\nSaved -> {args.out}")

if __name__ == "__main__":
    main()
