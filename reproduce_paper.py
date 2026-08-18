"""reproduce_paper.py - reproduce Table 1 under 5-fold stratified CV.

Classical SVM trains on the FULL training fold (not a 300-sample subset).
QSVM uses the analytic angle-encoding fidelity kernel, matched at C=3.0.
All models share identical folds (seed 0).

    python reproduce_paper.py --features snake_cnn_features.npz
"""
import argparse, time, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="snake_cnn_features.npz")
ap.add_argument("--qubits", type=int, default=8)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--C", type=float, default=3.0)
ap.add_argument("--qsvm_train", type=int, default=250,
                help="QSVM train subsample per fold (paper protocol); 0 = full")
ap.add_argument("--qsvm_test", type=int, default=150)
a = ap.parse_args()

def fidelity_kernel(A, B):
    d = A[:, None, :] - B[None, :, :]
    return np.exp(np.log(np.clip(np.cos(d / 2.0) ** 2, 1e-300, None)).sum(2))

d = np.load(a.features)
X, y = np.asarray(d["X"], float), np.asarray(d["y"]).astype(int)
print(f"Loaded {X.shape[0]} images x {X.shape[1]} features")
p = float((y == 1).mean())
print(f"Majority-class baseline: {max(p,1-p)*100:.1f}%  "
      f"(F1 {2*p/(p+1)*100:.1f}%)\n")

cv = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=0)
res = {k: [] for k in ["svm_full", "svm_f1", "qsvm", "qsvm_f1", "svm8"]}
t0 = time.time()

for k, (tr, te) in enumerate(cv.split(X, y), 1):
    sc = StandardScaler().fit(X[tr])
    Xtr_s, Xte_s = sc.transform(X[tr]), sc.transform(X[te])
    svm = SVC(kernel="rbf", C=a.C, gamma="scale", random_state=0).fit(Xtr_s, y[tr])
    pc = svm.predict(Xte_s)
    res["svm_full"].append(accuracy_score(y[te], pc))
    res["svm_f1"].append(f1_score(y[te], pc, zero_division=0))

    rs = np.random.RandomState(k)
    qtr = rs.choice(tr, a.qsvm_train, replace=False) if a.qsvm_train and len(tr) > a.qsvm_train else tr
    qte = rs.choice(te, a.qsvm_test, replace=False) if a.qsvm_test and len(te) > a.qsvm_test else te
    sc2 = StandardScaler().fit(X[qtr])
    A2, B2 = sc2.transform(X[qtr]), sc2.transform(X[qte])
    pca = PCA(n_components=a.qubits, random_state=0).fit(A2)
    A2, B2 = pca.transform(A2), pca.transform(B2)
    mm = MinMaxScaler((0, np.pi)).fit(A2)
    A2 = np.clip(mm.transform(A2), 0, np.pi)
    B2 = np.clip(mm.transform(B2), 0, np.pi)
    Ktr = fidelity_kernel(A2, A2)
    Kte = fidelity_kernel(B2, A2)
    qk = SVC(kernel="precomputed", C=a.C, random_state=0).fit(Ktr, y[qtr])
    pq = qk.predict(Kte)
    res["qsvm"].append(accuracy_score(y[qte], pq))
    res["qsvm_f1"].append(f1_score(y[qte], pq, zero_division=0))

    c8 = SVC(kernel="rbf", C=a.C, gamma="scale", random_state=0).fit(A2, y[qtr])
    res["svm8"].append(accuracy_score(y[qte], c8.predict(B2)))
    print(f"  fold {k}/{a.folds}  SVM {res['svm_full'][-1]*100:.1f}%  "
          f"QSVM {res['qsvm'][-1]*100:.1f}%  ({time.time()-t0:.0f}s)")

def stat(v):
    return np.mean(v) * 100, np.std(v) * 100

print("\n" + "=" * 56)
print("TABLE 1  (5-fold stratified CV, mean +/- s.d.)")
print("=" * 56)
m, s = stat(res["svm_full"]); f, _ = stat(res["svm_f1"])
print(f"  Classical SVM (full 1280)   {m:.1f} +/- {s:.1f}%   F1 {f:.1f}%")
m, s = stat(res["qsvm"]); f, _ = stat(res["qsvm_f1"])
print(f"  QSVM ({a.qubits}q, C={a.C})           {m:.1f} +/- {s:.1f}%   F1 {f:.1f}%")
m, s = stat(res["svm8"])
print(f"  Classical SVM (8 PCA feats) {m:.1f} +/- {s:.1f}%   [matched control]")
print("=" * 56)
print(f"  total time {time.time()-t0:.0f}s")
