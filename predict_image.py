import argparse
import csv
import os
import numpy as np
import joblib
from features import get_extractor
from serialize import rebuild

SAFETY = (
    "\n  !!  SAFETY  !!  Experimental classifier - it WILL make mistakes.\n"
    "  NEVER use it to decide whether a real snake is safe to approach.\n"
    "  Treat every unidentified snake as dangerous; contact professionals.\n"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", choices=["vqc", "qcnn", "svm"], default="vqc")
    ap.add_argument("--bundle", default="snake_model.joblib")
    args = ap.parse_args()

    if not os.path.exists(args.bundle):
        raise SystemExit(f"No trained model at {args.bundle} - run train_from_images.py first.")
    if not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    b = joblib.load(args.bundle)
    feat = get_extractor(b["extractor"])(args.image).reshape(1, -1)
    feat_s = b["scaler"].transform(feat)

    if args.model == "svm":
        proba = b["svm"].predict_proba(feat_s)[0]
        idx = int(np.argmax(proba)); confidence = float(proba[idx]) * 100
    else:
        state = b["vqc_state"] if args.model == "vqc" else b["qcnn_state"]
        if state is None:
            raise SystemExit(f"'{args.model}' was not trained in this bundle.")
        feat_q = b["pca"].transform(feat_s) if b.get("pca") is not None else feat_s
        model = rebuild(state)
        score = float(model.decision_function(feat_q)[0])
        idx = int(score > 0); confidence = abs(score) * 100

    label = b["classes"][idx]
    print("\n" + "=" * 52)
    print(f"  Photo   : {os.path.basename(args.image)}")
    extra = f"  ({b['n_qubits']} qubits, {b['encoding']} encoding)" if args.model != "svm" else ""
    print(f"  Model   : {args.model.upper()}" + extra)
    print(f"  VERDICT : {label}   (confidence ~{confidence:.0f}%)")
    print("=" * 52)
    print(SAFETY)

    with open("predictions_log.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["image", "model", "verdict", "confidence_pct"])
        w.writerow([args.image, args.model, label, round(confidence, 1)])

if __name__ == "__main__":
    main()
