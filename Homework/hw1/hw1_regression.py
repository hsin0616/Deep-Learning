import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler



# ===== Dealing with the data =====
df = pd.read_csv("energy_efficiency_data.csv")
print("Shape of the original data: ", df.shape)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["Orientation", "Glazing Area Distribution"])
X = df_encoded.drop(columns=["Heating Load", "Cooling Load"]).values
y = df_encoded["Heating Load"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True
)
print("After one-hot encoding, Training data shape:", X_train.shape, "| Test data shape:", X_test.shape)


def he_init(layer_dims, seed=42):
    rng = np.random.default_rng(seed)
    params = {}
    L = len(layer_dims) - 1
    for l in range(1, L+1):
        fan_in = layer_dims[l-1]
        fan_out = layer_dims[l]
        # He initialization for ReLU layers (fine for the last linear layer)
        params[f"W{l}"] = rng.normal(0.0, np.sqrt(2.0/fan_in), size=(fan_out, fan_in))
        params[f"b{l}"] = np.zeros((fan_out, 1), dtype=float)
    return params


X_train_np = X_train.T.astype(np.float64)  # (16, Ntrain)
y_train_np = y_train.reshape(1, -1).astype(np.float64)  # (1, Ntrain)
X_test_np  = X_test.T.astype(np.float64)
y_test_np  = y_test.reshape(1, -1).astype(np.float64)

# Standardize
mu = X_train_np.mean(axis=1, keepdims=True)
sigma = X_train_np.std(axis=1, keepdims=True) + 1e-8
X_train_np = (X_train_np - mu) / sigma
X_test_np  = (X_test_np  - mu) / sigma


# activation
def relu(Z):                return np.maximum(0, Z)
def relu_derivative(Z):     return (Z > 0).astype(float)
def sigmoid(Z):             return 1 / (1 + np.exp(-Z))
def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def tanh(Z):                return np.tanh(Z)
def tanh_derivative(Z):     return 1 - np.tanh(Z) ** 2
def linear(Z):    return Z
def linear_derivative(Z):   return np.ones_like(Z)


# forward step for a single layer
def layer_forward(A_prev, W, b, activation="relu"):
    Z = W @ A_prev + b

    if activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "tanh":
        A = tanh(Z)
    elif activation == "linear":
        A = linear(Z)
    else:
        raise ValueError("Unsupported activation")

    cache = (A_prev, W, b, Z, activation)
    return A, cache


# backward step for a single layer
def layer_backward(dA, cache):
    A_prev, W, b, Z, activation = cache
    m = A_prev.shape[1]  # batch size

    if activation == "relu":
        dZ = dA * relu_derivative(Z)
    elif activation == "sigmoid":
        dZ = dA * sigmoid_derivative(Z)
    elif activation == "tanh":
        dZ = dA * tanh_derivative(Z)
    elif activation == "linear":
        dZ = dA * linear_derivative(Z)
    else:
        raise ValueError("Unsupported activation")

    # Parameter gradients
    dW = (1 / m) * dZ @ A_prev.T
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


# === Build nn architecture ===
# ----- Define the architecture -----
layer_dims = [16, 64, 32, 1]                 # [n_in, h1, h2, n_out]
activations = ["relu", "relu", "linear"]     # one per affine layer

# (re)initialize parameters for this arch
params = he_init(layer_dims, seed=42)


# ----- Forward propagation across all layers -----
def model_forward(X, params, layer_dims, activations):
    assert len(activations) == len(layer_dims) - 1, "One activation per affine layer."
    A = X
    caches = []
    L = len(layer_dims) - 1   # number of affine layers

    for l in range(1, L + 1):
        A, cache = layer_forward(
            A, params[f"W{l}"], params[f"b{l}"], activation=activations[l-1]
        )
        caches.append(cache)

    y_hat = A   # final activation output (linear for regression)
    return y_hat, caches


# backward propagation
def mse_loss(y_hat, y):
    m = y.shape[1]
    return (1.0 / m) * np.sum((y_hat - y) ** 2)


def rmse(y_hat, y):    return np.sqrt(((y_hat - y)**2).mean())


def mse_grad(y_hat, y):
    """
    dL/d(y_hat) for MSE, shape (1, m)
    """
    m = y.shape[1]
    return (2.0 / m) * (y_hat - y)


# ----- Full network backward pass -----
def model_backward(y_hat, y, caches):
    grads = {}
    L = len(caches)
    dA = mse_grad(y_hat, y)  # derivative wrt final activation A_L (equals y_hat)

    # Backprop through the last layer first (its cache is caches[-1])
    dA_prev, dW, db = layer_backward(dA, caches[-1])
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db

    # Hidden layers (L-1 ... 1)
    for l in reversed(range(1, L)):
        dA_prev, dW, db = layer_backward(dA_prev, caches[l-1])
        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db

    return grads


def update_parameters(params, grads, learning_rate):
    L = len(params) // 2  # number of layers
    for l in range(1, L + 1):
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return params


def evaluate(X, Y, params, layer_dims, activations):
    """Compute MSE on a dataset."""
    y_hat, _ = model_forward(X, params, layer_dims, activations)
    return mse_loss(y_hat, Y)


def train_with_tracking(X_train, Y_train, X_test, Y_test, layer_dims, activations,
                        epochs=1000, learning_rate=0.01, print_every=100):
    params = he_init(layer_dims)
    train_losses, test_losses = [], []

    for i in range(epochs):
        # Forward
        y_hat, caches = model_forward(X_train, params, layer_dims, activations)
        # loss = mse_loss(y_hat, Y_train)
        loss = rmse(y_hat, Y_train)

        # Backward
        grads = model_backward(y_hat, Y_train, caches)

        # Update
        params = update_parameters(params, grads, learning_rate)

        # Track loss
        train_losses.append(loss)
        # test_losses.append(evaluate(X_test, Y_test, params, layer_dims, activations))
        y_hat_test, _ = model_forward(X_test, params, layer_dims, activations)
        test_losses.append(rmse(y_hat_test, Y_test))

        # Print
        if (i+1) % print_every == 0 or i == 0:
            print(f"Epoch {i+1:4d} | Train Loss={train_losses[-1]:.4f} | Test Loss={test_losses[-1]:.4f}")

    return params, train_losses, test_losses


# ---- Run training ----
params, train_losses, test_losses = train_with_tracking(
    X_train_np, y_train_np,
    X_test_np, y_test_np,
    layer_dims, activations,
    epochs=1000, learning_rate=0.01, print_every=100
)

# ---- Plot ----
plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Training Loss (RMSE)")
plt.plot(test_losses, label="Test Loss (RMSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()


# --- TRAIN ---
yhat_tr, _ = model_forward(X_train_np, params, layer_dims, activations)
ord_tr = np.argsort(y_train_np.ravel())
plt.figure(figsize=(8,4))
plt.plot(y_train_np.ravel()[ord_tr], label="label")
plt.plot(yhat_tr.ravel()[ord_tr],  label="predict")
plt.title(f"prediction for training data  |  RMSE={rmse(yhat_tr,y_train_np):.3f}")
plt.xlabel("#th case (sorted by label)"); plt.ylabel("heating load")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# --- TEST ---
yhat_te, _ = model_forward(X_test_np, params, layer_dims, activations)
ord_te = np.argsort(y_test_np.ravel())
plt.figure(figsize=(8,4))
plt.plot(y_test_np.ravel()[ord_te], label="label")
plt.plot(yhat_te.ravel()[ord_te],  label="predict")
plt.title(f"prediction for test data  |  RMSE={rmse(yhat_te,y_test_np):.3f}")
plt.xlabel("#th case (sorted by label)"); plt.ylabel("heating load")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


# --- forward wrapper ---
def plot_training_results(params, layer_dims, activations,
                          X_train, y_train, X_test, y_test,
                          train_losses,
                          arch_str="16-64-32-1",
                          selected_features=None,
                          y_mu=None, y_std=None,
                          title_curve="training curve"):
    # 1) Predictions
    yhat_train, _ = model_forward(X_train, params, layer_dims, activations)
    yhat_test,  _ = model_forward(X_test,  params, layer_dims, activations)

    # 2) Invert target standardization if provided
    if (y_mu is not None) and (y_std is not None):
        yhat_train_plot = yhat_train * y_std + y_mu
        yhat_test_plot  = yhat_test  * y_std + y_mu
        y_train_plot    = y_train    * y_std + y_mu
        y_test_plot     = y_test     * y_std + y_mu
    else:
        yhat_train_plot = yhat_train
        yhat_test_plot  = yhat_test
        y_train_plot    = y_train
        y_test_plot     = y_test

    # 3) Errors (RMSE)
    train_erms = rmse(yhat_train_plot, y_train_plot)
    test_erms  = rmse(yhat_test_plot,  y_test_plot)

    # 4) Figure layout
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.4], height_ratios=[1, 1], wspace=0.28, hspace=0.25)

    # (A) Report box (top-left)
    ax0 = plt.subplot(gs[0, 0]); ax0.axis("off")
    sel_str = str(list(selected_features)) if selected_features is not None else "[all]"
    lines = [
        rf"$\bf{{Network\ architecture}}$: {arch_str}",
        rf"$\bf{{Selected\ features}}$: {sel_str}",
        rf"$\bf{{Training}}\ E_{{RMS}}$: {train_erms:.5f}",
        rf"$\bf{{Test}}\ E_{{RMS}}$: {test_erms:.5f}",
    ]
    text = "\n".join(lines)
    ax0.text(0.0, 0.95, text, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.0), fontsize=11)

    # (B) Training curve (top-right)
    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(train_losses)
    ax1.set_title(title_curve)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (RMSE)")
    ax1.grid(True)

    # (C) Prediction for test data (bottom-left)
    ax2 = plt.subplot(gs[1, 0])
    # plot as sequences by case index; convert to 1D
    ax2.plot(y_test_plot.flatten(order="F"), label="label")
    ax2.plot(yhat_test_plot.flatten(order="F"), label="predict")
    ax2.set_title("prediction for test data")
    ax2.set_xlabel("#th case")
    ax2.set_ylabel("heating load")
    ax2.legend()
    ax2.grid(True)

    # (D) Prediction for training data (bottom-right)
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(y_train_plot.flatten(order="F"), label="label")
    ax3.plot(yhat_train_plot.flatten(order="F"), label="predict")
    ax3.set_title("prediction for training data")
    ax3.set_xlabel("#th case")
    ax3.set_ylabel("heating load")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return {"train_RMSE": train_erms, "test_RMSE": test_erms}


metrics = plot_training_results(
    params, layer_dims, activations,
    X_train_np, y_train_np,
    X_test_np,  y_test_np,
    train_losses,
    arch_str="16-64-32-1",
    selected_features=list(range(16))
)
print(metrics)


def summarize_arch(layer_dims, activations):
    print("Architecture:")
    print(f" Input: {layer_dims[0]}")
    hidden = layer_dims[1:-1]
    print(f" Hidden layers ({len(hidden)}): {hidden}")
    print(f" Output: {layer_dims[-1]}")
    print(f" Activations: {activations}")

    # parameter count
    total = 0
    for l in range(1, len(layer_dims)):
        fan_in, fan_out = layer_dims[l-1], layer_dims[l]
        nW, nb = fan_in*fan_out, fan_out
        total += nW + nb
        print(f"  Layer {l}: W({fan_out}x{fan_in})={nW}, b({fan_out})={nb}  -> {nW+nb}")
    print(f" Total parameters: {total}")

summarize_arch(layer_dims, activations)


# --- 1. Compute the importance of each feature (F-test) ---
feature_names = df_encoded.drop(columns=["Heating Load", "Cooling Load"]).columns
X_all = df_encoded.drop(columns=["Heating Load", "Cooling Load"]).values
y_all = df_encoded["Heating Load"].values

F, pval = f_regression(X_all, y_all)
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "F_score": F,
    "p_value": pval
}).sort_values(by="F_score", ascending=False)

print("\n=== Feature Importance Ranking (F-test) ===")
print(feature_importance)

# --- 2. k important features ---
topk = 8
selected_features = feature_importance["Feature"].head(topk).values
print("\nSelected top features:", selected_features)

# --- 3. restruct and split the data based on the features ---
X_sel = df_encoded[selected_features].values
y_sel = df_encoded["Heating Load"].values

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_sel, y_sel, test_size=0.25, random_state=42, shuffle=True
)

# --- 4. standardization ---
X_train_sel = np.asarray(X_train_sel, dtype=np.float64)
X_test_sel  = np.asarray(X_test_sel,  dtype=np.float64)

if X_train_sel.ndim == 1:  # topk=1 時會變成一維
    X_train_sel = X_train_sel.reshape(-1, 1)
    X_test_sel  = X_test_sel.reshape(-1, 1)

scaler = StandardScaler()
X_train_sel = scaler.fit_transform(X_train_sel)
X_test_sel  = scaler.transform(X_test_sel)

# --- 5. input format (features, samples) ---
X_train_sel = X_train_sel.T
X_test_sel  = X_test_sel.T
y_train_sel = y_train_sel.reshape(1, -1).astype(np.float64)
y_test_sel  = y_test_sel.reshape(1, -1).astype(np.float64)

# --- 6. retrain the model ---
params_sel, train_losses_sel, test_losses_sel = train_with_tracking(
    X_train_sel, y_train_sel,
    X_test_sel, y_test_sel,
    layer_dims=[topk, 64, 32, 1],
    activations=["relu", "relu", "linear"],
    epochs=1000, learning_rate=0.01, print_every=100
)

# --- 7. plot ---
plot_training_results(
    params_sel, [topk, 64, 32, 1], ["relu", "relu", "linear"],
    X_train_sel, y_train_sel, X_test_sel, y_test_sel,
    train_losses_sel,
    arch_str=f"{topk}-64-32-1",
    selected_features=list(selected_features)
)
