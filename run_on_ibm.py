"""
run_on_ibm.py — send your snake-project circuits to a REAL IBM quantum computer.

WHAT THIS DOES
--------------
Runs your three quantum circuits (VQC, QCNN, QSVM kernel) on actual IBM
quantum hardware instead of a simulator. This is "hardware validation" — it
shows how real quantum noise affects your results.

HONEST EXPECTATION
------------------
Real IBM machines are noisy NISQ devices. Your accuracy will DROP compared to
the noiseless simulator. That is the expected, publishable finding.

ONE-TIME SETUP (do this first)
------------------------------
1. Make a free IBM Quantum account:  https://quantum.ibm.com/
2. Copy your API token from the dashboard (top-right, account settings).
3. Install the IBM plugin:
       pip install pennylane-qiskit qiskit-ibm-runtime
4. Paste your token where it says IBM_TOKEN below.

Then run:  python run_on_ibm.py
"""

import numpy as np
import pennylane as qml

# ======================================================================
# PASTE YOUR IBM TOKEN HERE (from https://quantum.ibm.com/ dashboard)
# ======================================================================
IBM_TOKEN = "PASTE_YOUR_TOKEN_HERE"

N_QUBITS = 4          # start small (4) on real hardware; noise grows with qubits
SHOTS = 1024          # number of measurement repetitions


# ----------------------------------------------------------------------
# Connect to IBM hardware
# ----------------------------------------------------------------------
def get_ibm_device(n_qubits):
    """Return a PennyLane device backed by a real IBM quantum computer."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise SystemExit(
            "Missing packages. Install them first:\n"
            "    pip install pennylane-qiskit qiskit-ibm-runtime")

    if IBM_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        raise SystemExit(
            "Please paste your IBM API token into IBM_TOKEN at the top of this file.\n"
            "Get it free at https://quantum.ibm.com/")

    # save credentials (only needed once, but harmless to repeat)
    QiskitRuntimeService.save_account(
        channel="ibm_quantum", token=IBM_TOKEN, overwrite=True)
    service = QiskitRuntimeService()

    # pick the least-busy real backend with enough qubits
    backend = service.least_busy(operational=True, simulator=False,
                                 min_num_qubits=n_qubits)
    print(f"Using IBM backend: {backend.name} ({backend.num_qubits} qubits)")

    # PennyLane device that runs on that backend
    return qml.device("qiskit.remote", wires=n_qubits, backend=backend,
                      shots=SHOTS)


# ----------------------------------------------------------------------
# THE THREE CIRCUITS (same structure as your project)
# ----------------------------------------------------------------------
def make_circuits(dev, n):
    """Build the VQC, QCNN, and QSVM-kernel circuits on the given device."""

    # ---- 1. VQC ----
    @qml.qnode(dev)
    def vqc(x, weights):
        qml.AmplitudeEmbedding(x, wires=range(n), normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(weights, wires=range(n))
        return qml.expval(qml.PauliZ(0))

    # ---- 2. QCNN ----
    CONV_P, POOL_P = 6, 3

    def conv(p, a, b):
        qml.RY(p[0], wires=a); qml.RY(p[1], wires=b)
        qml.CNOT(wires=[a, b])
        qml.RY(p[2], wires=a); qml.RY(p[3], wires=b)
        qml.CNOT(wires=[b, a])
        qml.RY(p[4], wires=a); qml.RY(p[5], wires=b)

    def pool(p, s, k):
        qml.CRZ(p[0], wires=[s, k]); qml.CRX(p[1], wires=[s, k]); qml.RZ(p[2], wires=k)

    stages, active = [], list(range(n))
    while len(active) > 1:
        cp = [(active[i], active[i + 1]) for i in range(0, len(active), 2)]
        pp = [(active[i + 1], active[i]) for i in range(0, len(active), 2)]
        stages.append((cp, pp))
        active = [active[i] for i in range(0, len(active), 2)]
    qcnn_nparams = sum(len(c) * CONV_P + len(p) * POOL_P for c, p in stages)

    @qml.qnode(dev)
    def qcnn(x, p):
        qml.AmplitudeEmbedding(x, wires=range(n), normalize=True, pad_with=0.0)
        i = 0
        for cp, pp in stages:
            for a, b in cp:
                conv(p[i:i + CONV_P], a, b); i += CONV_P
            for s, k in pp:
                pool(p[i:i + POOL_P], s, k); i += POOL_P
        return qml.expval(qml.PauliZ(0))

    # ---- 3. QSVM kernel ----
    @qml.qnode(dev)
    def qsvm_kernel(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n), rotation="Y")
        return qml.probs(wires=range(n))

    return vqc, qcnn, qsvm_kernel, qcnn_nparams


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    n = N_QUBITS
    print("=" * 60)
    print("Running snake-project circuits on REAL IBM hardware")
    print("=" * 60)

    dev = get_ibm_device(n)
    vqc, qcnn, qsvm_kernel, qcnn_np = make_circuits(dev, n)

    rng = np.random.default_rng(0)

    # example inputs (replace with your real PCA-reduced features)
    x = rng.random(2 ** n)
    x_angle = rng.uniform(0, np.pi, n)
    x2_angle = rng.uniform(0, np.pi, n)

    vqc_w = 0.1 * rng.standard_normal(qml.StronglyEntanglingLayers.shape(3, n))
    qcnn_p = 0.1 * rng.standard_normal(qcnn_np)

    print("\n[1/3] Running VQC on IBM hardware (this queues on a real machine)...")
    print("      VQC output ⟨Z₀⟩ =", float(vqc(x, vqc_w)))

    print("\n[2/3] Running QCNN on IBM hardware...")
    print("      QCNN output ⟨Z₀⟩ =", float(qcnn(x, qcnn_p)))

    print("\n[3/3] Running QSVM kernel on IBM hardware...")
    k = float(qsvm_kernel(x_angle, x2_angle)[0])
    print("      Kernel value K(A,B) =", k)

    print("\n" + "=" * 60)
    print("DONE. These ran on a REAL quantum computer (with noise).")
    print("Compare these to your noiseless simulator values — the")
    print("differences ARE the effect of real quantum hardware noise.")
    print("=" * 60)


if __name__ == "__main__":
    main()
