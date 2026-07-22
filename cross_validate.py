import time, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pennylane as qml
from pennylane import numpy as pnp

SEED = 0
N_FOLDS = 5
NQ = 8
QSVM_TRAIN = 250
QSVM_TEST = 150

def amp_prep(X, nq):
    X = np.asarray(X, float)
    cap = 2 ** nq
    if X.shape[1] > cap:
        X = X[:, :cap]
    elif X.shape[1] < cap:
        X = np.pad(X, ((0, 0), (0, cap - X.shape[1])))
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return pnp.array(X / n, requires_grad=False)

def train_vqc(Xtr, ytr, Xte, nq=8, layers=3, epochs=40, lr=0.05):
    dev = qml.device("default.qubit", wires=nq)
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def c(x, w):
        qml.AmplitudeEmbedding(x, wires=range(nq), normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(w, wires=range(nq))
        return qml.expval(qml.PauliZ(0))
    Xb = amp_prep(Xtr, nq)
    yb = pnp.array(2*np.asarray(ytr,float)-1, requires_grad=False)
    shape = qml.StronglyEntanglingLayers.shape(layers, nq)
    rng = np.random.default_rng(SEED)
    w = pnp.array(0.1*rng.standard_normal(shape), requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    def cost(w): return pnp.mean((c(Xb, w) - yb)**2)
    for _ in range(epochs):
        w, _ = opt.step_and_cost(cost, w)
    return (np.asarray(c(amp_prep(Xte, nq), w)).ravel() > 0).astype(int)

CONV_P, POOL_P = 6, 3
def conv(p, a, b):
    qml.RY(p[0], wires=a); qml.RY(p[1], wires=b); qml.CNOT(wires=[a,b])
    qml.RY(p[2], wires=a); qml.RY(p[3], wires=b); qml.CNOT(wires=[b,a])
    qml.RY(p[4], wires=a); qml.RY(p[5], wires=b)
def pool(p, s, k):
    qml.CRZ(p[0], wires=[s,k]); qml.CRX(p[1], wires=[s,k]); qml.RZ(p[2], wires=k)

def train_qcnn(Xtr, ytr, Xte, nq=8, epochs=45, lr=0.05):
    dev = qml.device("default.qubit", wires=nq)
    stages, active = [], list(range(nq))
    while len(active) > 1:
        cp = [(active[i], active[i+1]) for i in range(0, len(active), 2)]
        pp = [(active[i+1], active[i]) for i in range(0, len(active), 2)]
        stages.append((cp, pp)); active = [active[i] for i in range(0, len(active), 2)]
    npar = sum(len(cp)*CONV_P + len(pp)*POOL_P for cp, pp in stages)
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def c(x, p):
        qml.AmplitudeEmbedding(x, wires=range(nq), normalize=True, pad_with=0.0)
        i = 0
        for cp, pp in stages:
            for a, b in cp: conv(p[i:i+CONV_P], a, b); i += CONV_P
            for s, k in pp: pool(p[i:i+POOL_P], s, k); i += POOL_P
        return qml.expval(qml.PauliZ(0))
    Xb = amp_prep(Xtr, nq)
    yb = pnp.array(2*np.asarray(ytr,float)-1, requires_grad=False)
    rng = np.random.default_rng(SEED)
    p = pnp.array(0.1*rng.standard_normal(npar), requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    def cost(p): return pnp.mean((c(Xb, p) - yb)**2)
    for _ in range(epochs):
        p, _ = opt.step_and_cost(cost, p)
    return (np.asarray(c(amp_prep(Xte, nq), p)).ravel() > 0).astype(int)

def run_qsvm(Xtr_q, ytr, Xte_q, nq=8):
    dev = qml.device("default.qubit", wires=nq)
    @qml.qnode(dev)
    def circ(a, b):
        qml.AngleEmbedding(a, wires=range(nq), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(b, wires=range(nq), rotation="Y")
        return qml.probs(wires=range(nq))
    kern = lambda a, b: float(circ(a, b)[0])
    K_tr = np.array([[kern(a, b) for b in Xtr_q] for a in Xtr_q])
    K_te = np.array([[kern(a, b) for b in Xtr_q] for a in Xte_q])
    svm = SVC(kernel="precomputed", C=1.0, random_state=SEED).fit(K_tr, ytr)
    return svm.predict(K_te)

print("="*66)
print(f"{N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
print("="*66)
d = np.load("snake_cnn_features.npz")
X, y = d["X"], d["y"]
print(f"Dataset: {X.shape[0]} photos, {X.shape[1]} features")
print(f"Class balance: venomous={int(y.sum())}, non-venomous={int((y==0).sum())}")
print("This will take 40-90 minutes. Progress printed per fold.\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
res = {m: {"acc": [], "f1": []} for m in ["SVM", "VQC", "QCNN", "QSVM"]}

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    t0 = time.time()
    print(f"--- Fold {fold}/{N_FOLDS} ---", flush=True)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    p = SVC(kernel="rbf", C=3.0, gamma="scale", random_state=SEED).fit(Xtr_s, ytr).predict(Xte_s)
    res["SVM"]["acc"].append(accuracy_score(yte, p)); res["SVM"]["f1"].append(f1_score(yte, p, zero_division=0))
    print(f"  SVM   acc={res['SVM']['acc'][-1]*100:5.1f}%", flush=True)

    p = train_vqc(Xtr_s, ytr, Xte_s, nq=NQ)
    res["VQC"]["acc"].append(accuracy_score(yte, p)); res["VQC"]["f1"].append(f1_score(yte, p, zero_division=0))
    print(f"  VQC   acc={res['VQC']['acc'][-1]*100:5.1f}%", flush=True)

    p = train_qcnn(Xtr_s, ytr, Xte_s, nq=NQ)
    res["QCNN"]["acc"].append(accuracy_score(yte, p)); res["QCNN"]["f1"].append(f1_score(yte, p, zero_division=0))
    print(f"  QCNN  acc={res['QCNN']['acc'][-1]*100:5.1f}%", flush=True)

    rng = np.random.default_rng(SEED + fold)
    itr = rng.permutation(len(Xtr_s))[:QSVM_TRAIN]
    ite = rng.permutation(len(Xte_s))[:QSVM_TEST]
    pca = PCA(n_components=NQ, random_state=SEED).fit(Xtr_s[itr])
    qs = MinMaxScaler((0, np.pi)).fit(pca.transform(Xtr_s[itr]))
    Xtr_q = qs.transform(pca.transform(Xtr_s[itr]))
    Xte_q = qs.transform(pca.transform(Xte_s[ite]))
    p = run_qsvm(Xtr_q, ytr[itr], Xte_q, nq=NQ)
    res["QSVM"]["acc"].append(accuracy_score(yte[ite], p))
    res["QSVM"]["f1"].append(f1_score(yte[ite], p, zero_division=0))
    print(f"  QSVM  acc={res['QSVM']['acc'][-1]*100:5.1f}%   ({time.time()-t0:.0f}s for fold)\n", flush=True)

print("="*66)
print("FINAL RESULTS (mean +/- std over 5 folds)")
print("="*66)
print(f"{'Model':<8} {'Accuracy (%)':<22} {'F1 (%)':<22}")
print("-"*66)
for m in ["SVM", "QSVM", "QCNN", "VQC"]:
    a, f = np.array(res[m]["acc"])*100, np.array(res[m]["f1"])*100
    print(f"{m:<8} {a.mean():5.1f} +/- {a.std():4.1f}         {f.mean():5.1f} +/- {f.std():4.1f}")
print("="*66)
print("\nPUT THESE IN TABLE 1 OF YOUR PAPER:")
for m in ["SVM", "QSVM", "QCNN", "VQC"]:
    a, f = np.array(res[m]["acc"])*100, np.array(res[m]["f1"])*100
    print(f"  {m:<6} | {a.mean():.1f} +/- {a.std():.1f} | {f.mean():.1f} +/- {f.std():.1f}")
np.savez("cv_results.npz", **{f"{m}_{k}": np.array(v[k]) for m, v in res.items() for k in ("acc","f1")})
print("\nSaved raw fold results -> cv_results.npz")
