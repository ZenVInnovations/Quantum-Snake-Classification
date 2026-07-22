import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def _prep_features(X, n_qubits, encoding):
    X = np.asarray(X, dtype=float)
    if encoding == "amplitude":
        cap = 2 ** n_qubits
        if X.shape[1] > cap:
            X = X[:, :cap]
        elif X.shape[1] < cap:
            X = np.pad(X, ((0, 0), (0, cap - X.shape[1])))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    return pnp.array(X, requires_grad=False)


def _embed(x, n_qubits, encoding):
    if encoding == "amplitude":
        qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True, pad_with=0.0)
    else:
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")


class VariationalQuantumClassifier:
    def __init__(self, n_qubits=8, n_layers=3, encoding="amplitude", epochs=40, lr=0.05, seed=0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.loss_history_ = []

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(x, weights):
            _embed(x, n_qubits, encoding)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
        self._circuit = circuit

    def _init_weights(self):
        shape = qml.StronglyEntanglingLayers.shape(self.n_layers, self.n_qubits)
        rng = np.random.default_rng(self.seed)
        return pnp.array(0.1 * rng.standard_normal(shape), requires_grad=True)

    def fit(self, X, y):
        Xb = _prep_features(X, self.n_qubits, self.encoding)
        y_pm = pnp.array(2 * np.asarray(y, dtype=float) - 1.0, requires_grad=False)
        w = self._init_weights()
        opt = qml.AdamOptimizer(self.lr)
        def cost(w):
            return pnp.mean((self._circuit(Xb, w) - y_pm) ** 2)
        for _ in range(self.epochs):
            w, c = opt.step_and_cost(cost, w)
            self.loss_history_.append(float(c))
        self.weights_ = w
        return self

    def decision_function(self, X):
        Xb = _prep_features(X, self.n_qubits, self.encoding)
        return np.asarray(self._circuit(Xb, self.weights_)).ravel()

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


CONV_P = 6
POOL_P = 3


def _conv_block(p, wa, wb):
    qml.RY(p[0], wires=wa); qml.RY(p[1], wires=wb)
    qml.CNOT(wires=[wa, wb])
    qml.RY(p[2], wires=wa); qml.RY(p[3], wires=wb)
    qml.CNOT(wires=[wb, wa])
    qml.RY(p[4], wires=wa); qml.RY(p[5], wires=wb)


def _pool_block(p, source, sink):
    qml.CRZ(p[0], wires=[source, sink])
    qml.CRX(p[1], wires=[source, sink])
    qml.RZ(p[2], wires=sink)


class QuantumCNN:
    def __init__(self, n_qubits=8, encoding="amplitude", epochs=45, lr=0.05, seed=0):
        assert n_qubits >= 2 and (n_qubits & (n_qubits - 1)) == 0, "QuantumCNN needs a power-of-two qubit count (2, 4, 8, 16)."
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.loss_history_ = []
        self._stages = []
        active = list(range(n_qubits))
        while len(active) > 1:
            conv_pairs = [(active[i], active[i + 1]) for i in range(0, len(active), 2)]
            pool_pairs = [(active[i + 1], active[i]) for i in range(0, len(active), 2)]
            kept = [active[i] for i in range(0, len(active), 2)]
            self._stages.append((conv_pairs, pool_pairs))
            active = kept
        self._final_wire = active[0]
        self.n_params = sum(len(cp) * CONV_P + len(pp) * POOL_P for cp, pp in self._stages)

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(x, params):
            _embed(x, n_qubits, encoding)
            i = 0
            for conv_pairs, pool_pairs in self._stages:
                for (wa, wb) in conv_pairs:
                    _conv_block(params[i:i + CONV_P], wa, wb); i += CONV_P
                for (src, sink) in pool_pairs:
                    _pool_block(params[i:i + POOL_P], src, sink); i += POOL_P
            return qml.expval(qml.PauliZ(self._final_wire))
        self._circuit = circuit

    def fit(self, X, y):
        Xb = _prep_features(X, self.n_qubits, self.encoding)
        y_pm = pnp.array(2 * np.asarray(y, dtype=float) - 1.0, requires_grad=False)
        rng = np.random.default_rng(self.seed)
        p = pnp.array(0.1 * rng.standard_normal(self.n_params), requires_grad=True)
        opt = qml.AdamOptimizer(self.lr)
        def cost(p):
            return pnp.mean((self._circuit(Xb, p) - y_pm) ** 2)
        for _ in range(self.epochs):
            p, c = opt.step_and_cost(cost, p)
            self.loss_history_.append(float(c))
        self.params_ = p
        return self

    def decision_function(self, X):
        Xb = _prep_features(X, self.n_qubits, self.encoding)
        return np.asarray(self._circuit(Xb, self.params_)).ravel()

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
