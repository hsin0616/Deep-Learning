import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    filter_size: int = 3
    stride: int = 1
    l2_lambda: float = 0.0
    verbose: int = 1


# -------------------------
# Data
# -------------------------
def load_mnist_with_val(seed: int = 42, val_ratio: float = 5000 / 60000):
    """Load MNIST and split train into train/val."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_ratio, random_state=seed, stratify=y_train
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def preprocess_images(x: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] and expand channel dim -> (N, 28, 28, 1)."""
    x = x.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)
    return x


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -------------------------
# Model
# -------------------------
def build_cnn(cfg: TrainConfig) -> tf.keras.Model:
    """A simple CNN for MNIST with optional L2 regularization."""
    reg = tf.keras.regularizers.l2(cfg.l2_lambda) if cfg.l2_lambda > 0 else None

    inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="input")
    x = tf.keras.layers.Conv2D(
        32, (cfg.filter_size, cfg.filter_size),
        strides=(cfg.stride, cfg.stride),
        activation="relu",
        kernel_regularizer=reg,
        name="conv1",
    )(inputs)
    x = tf.keras.layers.Conv2D(
        64, (cfg.filter_size, cfg.filter_size),
        strides=(cfg.stride, cfg.stride),
        activation="relu",
        kernel_regularizer=reg,
        name="conv2",
    )(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        128, activation="relu", kernel_regularizer=reg, name="dense1"
    )(x)
    outputs = tf.keras.layers.Dense(
        10, activation="softmax", kernel_regularizer=reg, name="output"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------
# Training / Evaluation
# -------------------------
def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: TrainConfig,
) -> tf.keras.callbacks.History:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=cfg.verbose,
    )
    return history


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> Tuple[float, float]:
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    return test_loss, test_acc


# -------------------------
# Visualization
# -------------------------
def plot_learning_curves(history: tf.keras.callbacks.History, title_prefix: str = "") -> None:
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, h.get("accuracy", []), label="Train Acc")
    plt.plot(epochs, h.get("val_accuracy", []), label="Val Acc")
    plt.title(f"{title_prefix} Accuracy".strip())
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, h.get("loss", []), label="Train Loss")
    plt.plot(epochs, h.get("val_loss", []), label="Val Loss")
    plt.title(f"{title_prefix} Loss".strip())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_weight_histograms(model: tf.keras.Model, layer_names: List[str], bins: int = 60) -> None:
    plt.figure(figsize=(10, 10))
    for i, name in enumerate(layer_names, 1):
        w = model.get_layer(name).get_weights()
        if not w:
            continue
        weights = w[0].flatten()
        plt.subplot(2, 2, i)
        plt.hist(weights, bins=bins)
        plt.title(f"Weights: {name}")
        plt.xlabel("Value")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def get_predictions(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    probs = model.predict(x, verbose=0)
    return np.argmax(probs, axis=1)


def show_examples(
    x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, correct: bool = False, k: int = 8
) -> None:
    if correct:
        indices = np.where(y_pred == y_true)[0]
        title = "Correctly Classified Examples"
    else:
        indices = np.where(y_pred != y_true)[0]
        title = "Misclassified Examples"

    if len(indices) == 0:
        print("No examples found for this category.")
        return

    chosen = indices[:k]
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(chosen):
        plt.subplot(2, 4, i + 1)
        plt.imshow(x[idx].reshape(28, 28), cmap="gray")
        plt.title(f"Label:{y_true[idx]} Pred:{y_pred[idx]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(model: tf.keras.Model, x_img: np.ndarray, max_maps: int = 8) -> None:
    """Visualize first few feature maps for each conv layer."""
    conv_layers = [layer for layer in model.layers if "conv" in layer.name]
    if not conv_layers:
        print("No conv layers found.")
        return

    layer_outputs = [layer.output for layer in conv_layers]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

    # x_img should be (1, 28, 28, 1)
    activations = activation_model.predict(x_img, verbose=0)

    for layer, act in zip(conv_layers, activations):
        n_features = act.shape[-1]
        size = act.shape[1]
        n_show = min(n_features, max_maps)

        display_grid = np.zeros((size, size * n_show))
        for i in range(n_show):
            feature_map = act[0, :, :, i]
            fm = feature_map - feature_map.mean()
            fm = fm / (feature_map.std() + 1e-5)
            fm = fm * 64 + 128
            fm = np.clip(fm, 0, 255).astype("uint8")
            display_grid[:, i * size : (i + 1) * size] = fm

        scale = 1.0 / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(f"Feature maps: {layer.name}")
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap="gray")
        plt.axis("off")
        plt.show()


# -------------------------
# Experiments
# -------------------------
def run_single_experiment(
    cfg: TrainConfig,
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    plot: bool = True,
):
    model = build_cnn(cfg)

    train_ds = make_tf_dataset(x_train, y_train, cfg.batch_size, training=True)
    val_ds = make_tf_dataset(x_val, y_val, cfg.batch_size, training=False)
    test_ds = make_tf_dataset(x_test, y_test, cfg.batch_size, training=False)

    history = train_model(model, train_ds, val_ds, cfg)
    test_loss, test_acc = evaluate_model(model, test_ds)

    if plot:
        plot_learning_curves(history, title_prefix=f"(fs={cfg.filter_size}, s={cfg.stride}, l2={cfg.l2_lambda})")
        plot_weight_histograms(model, ["conv1", "conv2", "dense1", "output"])

    return model, history, (test_loss, test_acc)


def grid_search_stride_filter(
    base_cfg: TrainConfig,
    x_train, y_train, x_val, y_val, x_test, y_test,
    strides=(1, 2),
    filter_sizes=(3, 5),
    epochs=3,
) -> Dict[Tuple[int, int], float]:
    results = {}
    for stride in strides:
        for fs in filter_sizes:
            cfg = TrainConfig(**{**base_cfg.__dict__, "stride": stride, "filter_size": fs, "epochs": epochs})
            model, history, (_, test_acc) = run_single_experiment(
                cfg, x_train, y_train, x_val, y_val, x_test, y_test, plot=False
            )
            results[(stride, fs)] = test_acc
            print(f"[Grid] stride={stride}, filter_size={fs} => test_acc={test_acc:.4f}")
    return results


def l2_sweep(
    base_cfg: TrainConfig,
    x_train, y_train, x_val, y_val, x_test, y_test,
    lambdas=(0.0, 1e-5, 1e-4, 1e-3),
    epochs=8,
) -> Dict[float, Dict]:
    results = {}
    for lam in lambdas:
        cfg = TrainConfig(**{**base_cfg.__dict__, "l2_lambda": lam, "epochs": epochs})
        model, history, (_, test_acc) = run_single_experiment(
            cfg, x_train, y_train, x_val, y_val, x_test, y_test, plot=False
        )
        results[lam] = {"model": model, "history": history.history, "test_acc": test_acc}
        print(f"[L2] lambda={lam:<8} => test_acc={test_acc:.4f}")
    return results


def plot_l2_results(l2_results: Dict[float, Dict]) -> None:
    plt.figure(figsize=(10, 4))

    # Val accuracy
    plt.subplot(1, 2, 1)
    for lam, r in l2_results.items():
        plt.plot(r["history"]["val_accuracy"], label=f"val acc λ={lam}")
    plt.title("Validation Accuracy vs Epoch (L2 sweep)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Train loss
    plt.subplot(1, 2, 2)
    for lam, r in l2_results.items():
        plt.plot(r["history"]["loss"], label=f"train loss λ={lam}")
    plt.title("Training Loss vs Epoch (includes L2 penalty)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Weight hist comparison: dense1
    plt.figure(figsize=(10, 8))
    for i, (lam, r) in enumerate(l2_results.items(), 1):
        plt.subplot(2, 2, i)
        w = r["model"].get_layer("dense1").get_weights()[0].flatten()
        plt.hist(w, bins=60)
        plt.title(f"dense1 weights, λ={lam}")
        plt.xlabel("Value")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    cfg = TrainConfig(seed=42, epochs=10, batch_size=128, filter_size=3, stride=1, l2_lambda=0.0, verbose=1)
    set_seed(cfg.seed)

    # Load & preprocess
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_with_val(seed=cfg.seed)
    print("Raw shapes:", x_train.shape, x_val.shape, x_test.shape)

    x_train = preprocess_images(x_train)
    x_val = preprocess_images(x_val)
    x_test = preprocess_images(x_test)
    print("Processed shapes:", x_train.shape, x_val.shape, x_test.shape)

    # 1) Train baseline model + plots
    model, history, (test_loss, test_acc) = run_single_experiment(
        cfg, x_train, y_train, x_val, y_val, x_test, y_test, plot=True
    )
    print(f"[Baseline] Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # 2) Show correct/misclassified examples
    y_pred = get_predictions(model, x_test)
    show_examples(x_test, y_test, y_pred, correct=False, k=8)
    show_examples(x_test, y_test, y_pred, correct=True, k=8)

    # 3) Feature map visualization for one sample
    img = x_test[0:1]  # shape (1,28,28,1)
    label = y_test[0]
    pred = y_pred[0]
    print(f"[FeatureMap] label={label}, pred={pred}")
    visualize_feature_maps(model, img, max_maps=8)

    # 4) Grid search (stride/filter) quick test (3 epochs each)
    print("\n--- Grid Search (stride/filter) ---")
    grid_results = grid_search_stride_filter(
        cfg, x_train, y_train, x_val, y_val, x_test, y_test,
        strides=(1, 2), filter_sizes=(3, 5), epochs=3
    )
    print("Grid results:", grid_results)

    # 5) L2 sweep + plots
    print("\n--- L2 Sweep ---")
    l2_results = l2_sweep(
        cfg, x_train, y_train, x_val, y_val, x_test, y_test,
        lambdas=(0.0, 1e-5, 1e-4, 1e-3), epochs=8
    )
    plot_l2_results(l2_results)


if __name__ == "__main__":
    main()
