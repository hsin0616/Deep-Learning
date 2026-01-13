"""
hw2_cifar10.py

A maintainable CIFAR-10 training script:
- tf.data pipeline with (Pad+RandomCrop+Flip) augmentation
- Optional Cutout
- Optional MixUp (correct Beta(alpha, alpha))
- Small CNN (BN + Dropout + L2)
- CosineDecay learning rate + SGD(momentum, nesterov)
- EarlyStopping + ModelCheckpoint (.keras)
- Clean plotting: epoch curves + batch loss
- Visualizations: sample predictions, weight histograms, feature maps

Run:
  python hw2_cifar10.py
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    seed: int = 42

    # Data
    img_size: int = 32
    num_classes: int = 10
    val_size: int = 5000
    batch_size: int = 128

    # Augment
    use_cutout: bool = False
    cutout_length: int = 8

    # MixUp
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # Model
    l2_coef: float = 5e-4
    dropout: float = 0.35

    # Train
    epochs: int = 80
    base_lr: float = 0.05
    patience: int = 15
    ckpt_path: str = "best_cifar10.keras"

    # Plots / Viz
    show_predictions: bool = True
    show_weight_hists: bool = True
    show_feature_maps: bool = True
    verbose: int = 1


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# -------------------------
# Data loading & preprocessing
# -------------------------
def load_cifar10_with_val(cfg: Config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Make a val split (last val_size samples)
    x_val, y_val = x_train[-cfg.val_size:], y_train[-cfg.val_size:]
    x_train, y_train = x_train[:-cfg.val_size], y_train[:-cfg.val_size]

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def augment_basic(image, label, img_size: int):
    # Pad(+4) -> RandomCrop(32) -> RandomFlip
    image = tf.image.resize_with_crop_or_pad(image, img_size + 4, img_size + 4)
    image = tf.image.random_crop(image, [img_size, img_size, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label


def cutout(image: tf.Tensor, img_size: int, length: int = 8) -> tf.Tensor:
    """Cutout: random square mask set to zero."""
    h = img_size
    w = img_size
    y = tf.random.uniform([], 0, h, dtype=tf.int32)
    x = tf.random.uniform([], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(y - length // 2, 0, h)
    y2 = tf.clip_by_value(y + length // 2, 0, h)
    x1 = tf.clip_by_value(x - length // 2, 0, w)
    x2 = tf.clip_by_value(x + length // 2, 0, w)

    mask = tf.ones((y2 - y1, x2 - x1, 3), dtype=image.dtype)
    paddings = [[y1, h - y2], [x1, w - x2], [0, 0]]
    mask = tf.pad(mask, paddings, constant_values=0.0)
    return image * (1.0 - mask)


def augment_pipeline(image, label, cfg: Config):
    image, label = augment_basic(image, label, cfg.img_size)
    if cfg.use_cutout:
        image = cutout(image, cfg.img_size, cfg.cutout_length)
    return image, label


def make_dataset(x, y, cfg: Config, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=min(len(x), 50000), reshuffle_each_iteration=True)
        ds = ds.map(lambda im, la: augment_pipeline(im, la, cfg),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -------------------------
# MixUp (correct Beta(alpha, alpha))
# -------------------------
def to_one_hot(y: tf.Tensor, num_classes: int) -> tf.Tensor:
    y = tf.squeeze(tf.cast(y, tf.int32), axis=1)  # (B,)
    return tf.one_hot(y, num_classes)             # (B, C)


def sample_beta(alpha: float) -> tf.Tensor:
    """Sample lam ~ Beta(alpha, alpha) using Gamma trick."""
    # x ~ Gamma(alpha,1), y ~ Gamma(alpha,1), lam = x/(x+y)
    x = tf.random.gamma(shape=[1], alpha=alpha, beta=1.0)
    y = tf.random.gamma(shape=[1], alpha=alpha, beta=1.0)
    lam = x / (x + y)
    return tf.squeeze(lam)  # scalar


def mixup_batch(images: tf.Tensor, labels_oh: tf.Tensor, alpha: float) -> Tuple[tf.Tensor, tf.Tensor]:
    lam = sample_beta(alpha)
    idx = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    mixed_x = lam * images + (1.0 - lam) * tf.gather(images, idx)
    mixed_y = lam * labels_oh + (1.0 - lam) * tf.gather(labels_oh, idx)
    return mixed_x, mixed_y


def apply_mixup(ds: tf.data.Dataset, cfg: Config) -> tf.data.Dataset:
    def _map(images, labels):
        labels_oh = to_one_hot(labels, cfg.num_classes)
        return mixup_batch(images, labels_oh, cfg.mixup_alpha)
    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


# -------------------------
# Model
# -------------------------
def build_model(cfg: Config) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(cfg.l2_coef) if (cfg.l2_coef and cfg.l2_coef > 0) else None
    L = tf.keras.layers

    inputs = L.Input(shape=(cfg.img_size, cfg.img_size, 3), name="input")
    x = inputs

    # Block 1
    x = L.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=reg, name="conv1")(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=reg, name="conv2")(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.MaxPooling2D()(x)
    x = L.Dropout(cfg.dropout)(x)

    # Block 2
    x = L.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=reg, name="conv3")(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=reg, name="conv4")(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.MaxPooling2D()(x)
    x = L.Dropout(cfg.dropout)(x)

    # Block 3
    x = L.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=reg, name="conv5")(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=reg, name="conv6")(x)
    x = L.BatchNormalization()(x); x = L.ReLU()(x)

    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(cfg.dropout)(x)

    x = L.Dense(256, kernel_regularizer=reg, name="dense1")(x)
    x = L.ReLU()(x)
    x = L.Dropout(cfg.dropout)(x)

    outputs = L.Dense(cfg.num_classes, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="SmallCNN_CIFAR10")


def compile_model(model: tf.keras.Model, cfg: Config, steps_per_epoch: int) -> None:
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.base_lr,
        decay_steps=steps_per_epoch * cfg.epochs
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=0.9, nesterov=True
    )

    if cfg.use_mixup:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])


# -------------------------
# Callbacks
# -------------------------
class BatchLossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses: List[float] = []

    def on_train_batch_end(self, batch, logs=None):
        if logs and "loss" in logs:
            self.batch_losses.append(float(logs["loss"]))


def make_callbacks(cfg: Config) -> Tuple[List[tf.keras.callbacks.Callback], BatchLossHistory]:
    batch_cb = BatchLossHistory()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            cfg.ckpt_path, monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=cfg.patience,
            restore_best_weights=True, verbose=1
        ),
        batch_cb
    ]
    return callbacks, batch_cb


# -------------------------
# Plotting & Visualization
# -------------------------
def plot_training(history: tf.keras.callbacks.History, batch_losses: List[float]) -> None:
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    plt.figure(figsize=(14, 4))

    # Loss (epoch)
    plt.subplot(1, 3, 1)
    plt.plot(epochs, h["loss"], label="Train Loss")
    plt.plot(epochs, h["val_loss"], label="Val Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Accuracy (epoch)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, h["accuracy"], label="Train Acc")
    plt.plot(epochs, h["val_accuracy"], label="Val Acc")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # Loss (iteration)
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, label="Batch Loss")
    plt.title("Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_sample_predictions(model: tf.keras.Model, test_ds: tf.data.Dataset, n: int = 12) -> None:
    # Take a batch
    images, labels = next(iter(test_ds.unbatch().batch(max(n, 16))))
    preds = model.predict(images, verbose=0)
    pred_ids = tf.argmax(preds, axis=1).numpy()
    labels = labels.numpy().reshape(-1)

    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy())
        title = f"pred: {CLASS_NAMES[pred_ids[i]]}\ntrue: {CLASS_NAMES[int(labels[i])]}"
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_weight_histograms(model: tf.keras.Model,
                           layer_names: Tuple[str, ...] = ("conv1", "conv2", "dense1", "output"),
                           bins: int = 100) -> None:
    weights = []
    names = []
    for name in layer_names:
        try:
            layer = model.get_layer(name)
            w = layer.get_weights()
            if not w:
                continue
            kernel = w[0].flatten()
            weights.append(kernel)
            names.append(name)
        except ValueError:
            continue

    if not weights:
        print("No eligible layers found for weight histograms.")
        return

    rows, cols = 2, 2
    plt.figure(figsize=(10, 8))
    for i, (w, nm) in enumerate(zip(weights, names), start=1):
        plt.subplot(rows, cols, i)
        plt.hist(w, bins=bins)
        plt.title(f"Histogram: {nm}")
        plt.xlabel("Value")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(model: tf.keras.Model,
                           x_img: np.ndarray,
                           true_label: int,
                           max_layers: int = 2,
                           maps_per_layer: int = 6) -> None:
    """
    x_img: (1,32,32,3) single image batch
    """
    conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    conv_layers = conv_layers[:max_layers]

    if not conv_layers:
        print("No Conv2D layers found.")
        return

    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=[l.output for l in conv_layers]
    )

    acts = activation_model.predict(x_img, verbose=0)
    pred = np.argmax(model.predict(x_img, verbose=0), axis=1)[0]

    max_rows, max_cols = 2, 8
    max_slots = max_rows * max_cols
    tiles: List[Tuple[str, np.ndarray]] = [("input", x_img[0])]

    for li, a in enumerate(acts):
        k = min(maps_per_layer, a.shape[-1])
        for j in range(k):
            tiles.append((f"{conv_layers[li].name}[{j}]", a[0, :, :, j]))

    tiles = tiles[:max_slots]

    plt.figure(figsize=(max_cols * 2.2, max_rows * 2.2))
    for i, (name, arr) in enumerate(tiles, start=1):
        ax = plt.subplot(max_rows, max_cols, i)
        if name == "input":
            ax.imshow(arr)
            ax.set_title(
                f"true:{CLASS_NAMES[int(true_label)]}\npred:{CLASS_NAMES[int(pred)]}",
                fontsize=8
            )
        else:
            fm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            ax.imshow(fm, cmap="gray")
            ax.set_title(name, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# -------------------------
# Train / Eval
# -------------------------
def train_and_evaluate(cfg: Config):
    set_seed(cfg.seed)
    print("TF Version:", tf.__version__)
    print("Config:", cfg)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_with_val(cfg)
    print("Shapes:",
          "train", x_train.shape, y_train.shape,
          "val", x_val.shape, y_val.shape,
          "test", x_test.shape, y_test.shape)

    train_ds = make_dataset(x_train, y_train, cfg, training=True)
    val_ds = make_dataset(x_val, y_val, cfg, training=False)
    test_ds = make_dataset(x_test, y_test, cfg, training=False)

    # MixUp switches labels to one-hot; only apply to training set.
    if cfg.use_mixup:
        train_ds = apply_mixup(train_ds, cfg)

    steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())

    model = build_model(cfg)
    compile_model(model, cfg, steps_per_epoch)
    model.summary()

    callbacks, batch_cb = make_callbacks(cfg)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=cfg.verbose
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[Test] loss={test_loss:.4f}, acc={test_acc:.4f}")

    # Plot curves
    plot_training(history, batch_cb.batch_losses)

    # Visualizations
    if cfg.show_predictions:
        show_sample_predictions(model, test_ds, n=12)

    if cfg.show_weight_hists:
        plot_weight_histograms(model)

    if cfg.show_feature_maps:
        # Pick one sample from test set (numpy array for easy indexing)
        idx = 0
        x_img = x_test[idx:idx+1]  # (1,32,32,3)
        true_label = int(y_test[idx])
        visualize_feature_maps(model, x_img, true_label=true_label)

    return model, history, (test_loss, test_acc)


# -------------------------
# Main
# -------------------------
def main():
    cfg = Config(
        seed=42,
        use_cutout=False,
        use_mixup=False,
        mixup_alpha=0.2,
        l2_coef=5e-4,
        dropout=0.35,
        epochs=80,
        base_lr=0.05,
        patience=15,
        ckpt_path="best_cifar10.keras",
        verbose=1,
        show_predictions=True,
        show_weight_hists=True,
        show_feature_maps=True,
    )
    train_and_evaluate(cfg)


if __name__ == "__main__":
    main()
