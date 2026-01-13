import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)


# 0) Read the csv file and preprocessing
df = pd.read_csv("2025_ionosphere_data.csv")

X = df.iloc[:, :-1].values                                   # Features = all columns except last
y = df.iloc[:, -1].map({'g': 1, 'b': 0}).values.astype(int)  # Labels = last column; map 'g'->1, 'b'->0


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 1) Activations (forward)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0.0, z)


# 2) Activation derivatives
def d_sigmoid(a):     # expects a = sigmoid(z)
    return a * (1.0 - a)


def d_relu(z):        # uses pre-activation z
    return (z > 0).astype(z.dtype)


# 3) Initialize network params
#    He for ReLU, Xavier for Sigmoid
def init_layers(arch, seed=SEED):
    rng_local = np.random.default_rng(seed)
    params = {}
    for i, layer in enumerate(arch, start=1):
        n_in, n_out = layer["in"], layer["out"]
        act = layer["act"].lower()
        if act == "relu":
            W = rng_local.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        else:  # xavier for sigmoid
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = rng_local.uniform(-limit, limit, size=(n_in, n_out))
        b = np.zeros((1, n_out))
        params[f"W{i}"] = W
        params[f"b{i}"] = b
    return params


# 4) Forward propagation
def forward_propagation(X, params, arch):
    A = X
    cache = {"A0": X}
    for i, layer in enumerate(arch, start=1):
        Z = A @ params[f"W{i}"] + params[f"b{i}"]
        if layer["act"].lower() == "relu":
            A = relu(Z)
        elif layer["act"].lower() == "sigmoid":
            A = sigmoid(Z)
        else:
            raise ValueError(f"Unsupported activation {layer['act']}")
        cache[f"Z{i}"] = Z
        cache[f"A{i}"] = A
    return A, cache  # A is y_hat


# 5) Binary cross-entropy loss
def bce_loss(y_hat, y):
    y = y.reshape(-1, 1)
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


# 6) Backward propagation
def backward_propagation(y_hat, y, cache, params, arch):
    grads = {}
    N = y.shape[0]
    y = y.reshape(-1, 1)

    dA = None
    for i in reversed(range(1, len(arch) + 1)):
        A_prev = cache[f"A{i - 1}"]
        Z = cache[f"Z{i}"]
        W = params[f"W{i}"]
        act = arch[i - 1]["act"].lower()

        if i == len(arch):
            # Sigmoid + BCE → dZ = (y_hat - y)
            dZ = (y_hat - y)  # (N,1)
        else:
            if act == "relu":
                dZ = dA * d_relu(Z)
            elif act == "sigmoid":
                A_curr = cache[f"A{i}"]
                dZ = dA * d_sigmoid(A_curr)
            else:
                raise ValueError(f"Unsupported activation {act}")

        dW = (A_prev.T @ dZ) / N
        db = dZ.mean(axis=0, keepdims=True)
        dA = dZ @ W.T

        grads[f"dW{i}"] = dW
        grads[f"db{i}"] = db

    return grads


# 7) Parameter update (SGD)
def update(params, grads, arch, lr):
    for i in range(1, len(arch) + 1):
        params[f"W{i}"] -= lr * grads[f"dW{i}"]
        params[f"b{i}"] -= lr * grads[f"db{i}"]
    return params


# 8) Prediction & accuracy
def predict_proba(X, params, arch):
    y_hat, _ = forward_propagation(X, params, arch)
    return y_hat.ravel()

def accuracy(X, y, params, arch, threshold=0.5):
    p = predict_proba(X, params, arch)
    y_pred = (p >= threshold).astype(int)
    return (y_pred == y).mean()


# 9) Training loop (mini-batch SGD)
def train(
    X_tr, y_tr, X_te, y_te, arch,
    epochs=600, lr=0.01, batch_size=64, print_every=100, seed=SEED
):
    params = init_layers(arch, seed=seed)
    loss_hist, train_err_hist, test_err_hist = [], [], []

    N = X_tr.shape[0]

    for ep in range(1, epochs + 1):
        # Shuffle each epoch
        idx = np.random.default_rng(seed + ep).permutation(N)
        Xb, yb = X_tr[idx], y_tr[idx]

        # ---- Mini-batch loop (this is the "photo" SGD block) ----
        for start in range(0, N, batch_size):
            end = start + batch_size
            X_mb = Xb[start:end]
            y_mb = yb[start:end]
            if X_mb.shape[0] == 0:
                continue

            # Forward
            y_hat, cache = forward_propagation(X_mb, params, arch)
            # Backward
            grads = backward_propagation(y_hat, y_mb, cache, params, arch)
            # Update
            params = update(params, grads, arch, lr)

        # ---- Metrics at epoch end (on full sets) ----
        y_hat_full, _ = forward_propagation(X_tr, params, arch)
        loss = bce_loss(y_hat_full, y_tr)
        tr_err = 1 - accuracy(X_tr, y_tr, params, arch)
        te_err = 1 - accuracy(X_te, y_te, params, arch)

        loss_hist.append(loss)
        train_err_hist.append(tr_err)
        test_err_hist.append(te_err)

        if ep % print_every == 0 or ep == 1 or ep == epochs:
            print(f"epoch {ep:5d} | loss: {loss:.6f} | train err: {tr_err:.4f} | test err: {te_err:.4f}")

    return params, np.array(loss_hist), np.array(train_err_hist), np.array(test_err_hist)


# 10) Define architecture and run
#     (input dim inferred from training data)
in_dim = X_train.shape[1]
nn_architecture = [
    {"in": in_dim, "out": 32, "act": "relu"},
    {"in": 32, "out": 16, "act": "relu"},
    {"in": 16, "out": 1,  "act": "sigmoid"},
]

params, loss_hist, train_err_hist, test_err_hist = train(
    X_train, y_train, X_test, y_test,
    arch=nn_architecture,
    epochs=600,
    lr=0.01,
    batch_size=64,
    print_every=100,
    seed=SEED
)


# 11) Final report
tr_acc = 1 - train_err_hist[-1]
te_acc = 1 - test_err_hist[-1]
print(f"\nFinal Train Acc: {tr_acc:.4f} | Final Test Acc: {te_acc:.4f}")
print(f"\nTraining error rate: {1-tr_acc:.4f} | Test error rate: {1-te_acc:.4f}")


# 12) Plots
plt.figure()
plt.plot(np.arange(1, len(loss_hist)+1), loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Learning Curve (Loss)")
plt.tight_layout()

plt.figure()
plt.plot(np.arange(1, len(train_err_hist)+1), train_err_hist, label="Train Error")
plt.plot(np.arange(1, len(test_err_hist)+1), test_err_hist, label="Test Error")
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.title("Error Curves")
plt.legend()
plt.tight_layout()

plt.show()


# (c)
# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# =========================
# 0) Read the csv file and preprocessing
# =========================
df = pd.read_csv("2025_ionosphere_data.csv")

# Features = all columns except last
X = df.iloc[:, :-1].values

# Labels = last column; map 'g'->1, 'b'->0
y = df.iloc[:, -1].map({'g': 1, 'b': 0}).values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Helper: get latent (penultimate) activations
def get_latent(X, params, arch):
    A = X
    penultimate_A = None
    for i, layer in enumerate(arch, start=1):
        Z = A @ params[f"W{i}"] + params[f"b{i}"]
        if layer["act"].lower() == "relu":
            A = relu(Z)
        elif layer["act"].lower() == "sigmoid":
            A = sigmoid(Z)
        else:
            raise ValueError(f"Unsupported activation {layer['act']}")
        # Save when this is the penultimate layer
        if i == len(arch) - 1:
            penultimate_A = A.copy()
    return penultimate_A


# Train (mini-batch) + capture latent snapshots
def train_with_latent(
    X_tr, y_tr, X_te, y_te, arch,
    epochs=300, lr=0.01, batch_size=64, print_every=100,
    capture_epochs=(1, 5, 20, 100, 300), seed=SEED
):
    params = init_layers(arch, seed=seed)
    loss_hist, train_err_hist, test_err_hist = [], [], []
    latent_snapshots = {}   # epoch -> latent matrix on (subset of) train

    N = X_tr.shape[0]
    # choose a subset for plotting to keep figures readable
    max_points = min(500, N)
    plot_idx = np.random.default_rng(seed).choice(N, size=max_points, replace=False)

    for ep in range(1, epochs + 1):
        # Shuffle each epoch
        idx = np.random.default_rng(seed + ep).permutation(N)
        Xb, yb = X_tr[idx], y_tr[idx]

        for start in range(0, N, batch_size):
            end = start + batch_size
            X_mb = Xb[start:end]
            y_mb = yb[start:end]
            if X_mb.shape[0] == 0:
                continue
            # Forward
            y_hat, cache = forward_propagation(X_mb, params, arch)
            # Backward
            grads = backward_propagation(y_hat, y_mb, cache, params, arch)
            # Update
            params = update(params, grads, arch, lr)

        # metrics at epoch end (full sets)
        y_hat_full, _ = forward_propagation(X_tr, params, arch)
        loss = bce_loss(y_hat_full, y_tr)
        tr_err = 1 - accuracy(X_tr, y_tr, params, arch)
        te_err = 1 - accuracy(X_te, y_te, params, arch)

        loss_hist.append(loss)
        train_err_hist.append(tr_err)
        test_err_hist.append(te_err)

        if ep % print_every == 0 or ep == 1 or ep == epochs:
            print(f"epoch {ep:5d} | loss: {loss:.6f} | train err: {tr_err:.4f} | test err: {te_err:.4f}")

        # ---- capture latent at specified epochs ----
        if ep in capture_epochs:
            latent = get_latent(X_tr[plot_idx], params, arch)
            latent_snapshots[ep] = (latent, y_tr[plot_idx])

    return params, np.array(loss_hist), np.array(train_err_hist), np.array(test_err_hist), latent_snapshots


# different penultimate sizes
in_dim = X_train.shape[1]
penultimate_sizes = [2, 4, 8, 16]  # compare these
capture_epochs = (1, 5, 20, 100, 300)

results = {}  # width -> (params, loss_hist, train_err_hist, test_err_hist, latent_snaps)
for width in penultimate_sizes:
    print("\n" + "="*70)
    print(f"Training with penultimate width = {width}")
    print("="*70)
    nn_architecture = [
        {"in": in_dim, "out": 32, "act": "relu"},
        {"in": 32, "out": width, "act": "relu"},   # <-- penultimate layer (latent)
        {"in": width, "out": 1,  "act": "sigmoid"},
    ]
    res = train_with_latent(
        X_train, y_train, X_test, y_test, arch=nn_architecture,
        epochs=300, lr=0.01, batch_size=64, print_every=50,
        capture_epochs=capture_epochs, seed=SEED
    )
    results[width] = res


# Plot learning curves (one panel per width)
fig, axes = plt.subplots(1, len(penultimate_sizes), figsize=(4*len(penultimate_sizes), 3), sharey=True)
if len(penultimate_sizes) == 1:
    axes = [axes]
for ax, width in zip(axes, penultimate_sizes):
    _, loss_hist, train_err_hist, test_err_hist, _ = results[width]
    ax.plot(loss_hist, label='Loss')
    ax.plot(train_err_hist, label='Train Err')
    ax.plot(test_err_hist, label='Test Err')
    ax.set_title(f"Width={width}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
plt.tight_layout()


# Plot latent distributions over epochs
# For each width, create a figure with subplots for each captured epoch.
# If latent dim > 2, reduce to 2D with PCA for visualization.
for width in penultimate_sizes:
    _, _, _, _, latent_snaps = results[width]
    epochs_sorted = sorted(latent_snaps.keys())
    n_cols = len(epochs_sorted)
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 3))
    if n_cols == 1:
        axes = [axes]
    for j, ep in enumerate(epochs_sorted):
        latent, y_sub = latent_snaps[ep]
        # project to 2D if needed
        if latent.shape[1] == 2:
            latent_2d = latent
        else:
            pca = PCA(n_components=2, random_state=SEED)
            latent_2d = pca.fit_transform(latent)
        ax = axes[j]
        ax.scatter(latent_2d[y_sub==0, 0], latent_2d[y_sub==0, 1], s=10, alpha=0.7, label='b=0')
        ax.scatter(latent_2d[y_sub==1, 0], latent_2d[y_sub==1, 1], s=10, alpha=0.7, label='g=1')
        ax.set_title(f"Width={width} @ epoch {ep}")
        ax.set_xlabel("Latent-1 (2D proj)")
        ax.set_ylabel("Latent-2 (2D proj)")
        if j == 0:
            ax.legend(loc='best', fontsize=8)
    plt.tight_layout()

plt.show()
