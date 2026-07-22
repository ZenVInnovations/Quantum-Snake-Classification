import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

SEED = 0
N_FOLDS = 5

def ci95(vals):
    a = np.array(vals)
    if len(a) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))

def main():
    print("=" * 66)
    print("REVIEWER-REQUESTED CONTROL EXPERIMENTS")
    print("=" * 66)
    d = np.load("snake_cnn_features.npz")
    X, y = d["X"], d["y"]
    print(f"Dataset: {X.shape[0]} images, {X.shape[1]} features\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print("[A] Classical SVM vs PCA dimensionality (the key fairness control)")
    pca_dims = [4, 8, 16, 32, 64]
    resA = {k: [] for k in pca_dims}
    full = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        full.append(accuracy_score(
            y[te], SVC(kernel="rbf", C=3.0, gamma="scale",
                       random_state=SEED).fit(Xtr, y[tr]).predict(Xte)))
        for k in pca_dims:
            pca = PCA(n_components=k, random_state=SEED).fit(Xtr)
            Xtr_k, Xte_k = pca.transform(Xtr), pca.transform(Xte)
            resA[k].append(accuracy_score(
                y[te], SVC(kernel="rbf", C=3.0, gamma="scale",
                           random_state=SEED).fit(Xtr_k, y[tr]).predict(Xte_k)))
    print(f"    Classical SVM, full 1280 feats : "
          f"{np.mean(full)*100:5.1f} +/- {np.std(full)*100:.1f}%  "
          f"(95% CI +/-{ci95(full)*100:.1f})")
    for k in pca_dims:
        v = resA[k]
        print(f"    Classical SVM, {k:2d} PCA feats     : "
              f"{np.mean(v)*100:5.1f} +/- {np.std(v)*100:.1f}%  "
              f"(95% CI +/-{ci95(v)*100:.1f})")
    print("    ^ Compare the 8-PCA row to QSVM (65.5%).\n")

    print("[B] Classical SVM on the SAME 250-sample subsample QSVM used")
    resB_full, resB_pca8 = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        rng = np.random.default_rng(SEED + fold)
        itr = rng.permutation(len(tr))[:250]
        ite = rng.permutation(len(te))[:150]
        tr_s, te_s = tr[itr], te[ite]
        sc = StandardScaler().fit(X[tr_s])
        Xtr, Xte = sc.transform(X[tr_s]), sc.transform(X[te_s])
        resB_full.append(accuracy_score(
            y[te_s], SVC(kernel="rbf", C=3.0, random_state=SEED)
            .fit(Xtr, y[tr_s]).predict(Xte)))
        pca = PCA(n_components=8, random_state=SEED).fit(Xtr)
        resB_pca8.append(accuracy_score(
            y[te_s], SVC(kernel="rbf", C=3.0, random_state=SEED)
            .fit(pca.transform(Xtr), y[tr_s]).predict(pca.transform(Xte))))
    print(f"    Classical SVM, full feats, 250 samples : "
          f"{np.mean(resB_full)*100:5.1f} +/- {np.std(resB_full)*100:.1f}%")
    print(f"    Classical SVM, 8 PCA,     250 samples  : "
          f"{np.mean(resB_pca8)*100:5.1f} +/- {np.std(resB_pca8)*100:.1f}%")
    print("    ^ The 8-PCA/250-sample row is the TRUE match to QSVM.\n")

    print("[C] Classical MLP baseline (full features)")
    resC = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                            random_state=SEED)
        mlp.fit(Xtr, y[tr])
        resC.append(accuracy_score(y[te], mlp.predict(Xte)))
    print(f"    Classical MLP, full feats : "
          f"{np.mean(resC)*100:5.1f} +/- {np.std(resC)*100:.1f}%  "
          f"(95% CI +/-{ci95(resC)*100:.1f})\n")

    print("=" * 66)
    print("SUMMARY TABLE FOR THE PAPER")
    print("=" * 66)
    print(f"  Classical SVM (1280 feats) : {np.mean(full)*100:.1f} +/- {np.std(full)*100:.1f}%")
    print(f"  Classical SVM (8 PCA)      : {np.mean(resA[8])*100:.1f} +/- {np.std(resA[8])*100:.1f}%   <- KEY")
    print(f"  Classical MLP (1280 feats) : {np.mean(resC)*100:.1f} +/- {np.std(resC)*100:.1f}%")
    print(f"  QSVM (8 PCA, 250 samples)  : 65.5 +/- 1.9%   (from main experiment)")
    print()
    d8 = np.mean(resA[8]) * 100
    print(f"  If classical-PCA-8 ({d8:.1f}%) >> QSVM (65.5%): quantum adds a penalty.")
    print(f"  If classical-PCA-8 (~65%) close to QSVM: gap is mostly dimensionality.")
    np.savez("control_results.npz", full=full, pca8=resA[8], mlp=resC,
             svm_sub_full=resB_full, svm_sub_pca8=resB_pca8)
    print("\nSaved -> control_results.npz")

if __name__ == "__main__":
    main()
