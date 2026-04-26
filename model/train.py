"""
MIRA — Training script for retinal fundus disease classification.

Dataset: JSIEC-1000  (./samples/1000images/<class>/*.jpg)
Backbone: MobileNetV2 (ImageNet pretrained), transfer learning + fine-tune
Output:
  model/saved/best.keras     final Keras model
  model/saved/classes.json   {"index_to_label": [...]}
  model/saved/history.json   training metrics per epoch
  model/tfjs/                TensorFlow.js graph model for the browser

Setup (Windows, from project root D:/Code/PBL):
  python -m venv .venv
  .venv\\Scripts\\activate
  pip install -r model/requirements.txt
  python model/train.py

Tips:
  - First run: the script does a "head only" warm-up (frozen backbone, fast)
    followed by fine-tuning the top layers (slower, higher accuracy).
  - On CPU the warm-up phase takes ~3-5 min; fine-tune ~10-15 min.
  - To skip fine-tuning pass --no-finetune.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ───────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # D:/Code/PBL
DATA_DIR = ROOT / "samples" / "1000images"
SAVE_DIR = ROOT / "model" / "saved"
TFJS_DIR = ROOT / "model" / "tfjs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TFJS_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────────
# Hyperparameters
# ───────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH = 32
SEED = 42
HEAD_EPOCHS = 12
FINETUNE_EPOCHS = 12
HEAD_LR = 1e-3
FINETUNE_LR = 1e-5
DROPOUT = 0.4
LABEL_SMOOTHING = 0.05
FINETUNE_UNFREEZE_LAYERS = 30

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

BACKBONES = {
    "mobilenet": {
        "ctor": lambda shape: tf.keras.applications.MobileNetV2(
            input_shape=shape, include_top=False, weights="imagenet"
        ),
        "preprocess": "rescale_pm1",      # [-1, 1]
    },
    "efficientnet": {
        "ctor": lambda shape: tf.keras.applications.EfficientNetB0(
            input_shape=shape, include_top=False, weights="imagenet"
        ),
        "preprocess": "passthrough",      # EfficientNet has its own internal scaling
    },
    "efficientnet_b1": {
        "ctor": lambda shape: tf.keras.applications.EfficientNetB1(
            input_shape=shape, include_top=False, weights="imagenet"
        ),
        "preprocess": "passthrough",
    },
}


# ───────────────────────────────────────────────────────────────
# JSON helper — Keras / NumPy emit float32, which json can't dump
# ───────────────────────────────────────────────────────────────
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ───────────────────────────────────────────────────────────────
# Data discovery
# ───────────────────────────────────────────────────────────────
def discover_files(data_dir: Path):
    """Walk the dataset directory, return (paths, labels, class_names)."""
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    paths, labels = [], []
    for idx, d in enumerate(class_dirs):
        for f in d.iterdir():
            if f.suffix.lower() in EXTS:
                paths.append(str(f))
                labels.append(idx)
    return np.array(paths), np.array(labels), class_names


def stratified_splits(paths, labels, val_frac=0.10, test_frac=0.10):
    """Stratified train/val/test split. Falls back if a class has too few items."""
    # Force at least one sample per class in val and test where possible.
    counts = Counter(labels.tolist())
    enough = all(c >= 5 for c in counts.values())

    if enough:
        # standard stratified split
        x_tv, x_test, y_tv, y_test = train_test_split(
            paths, labels, test_size=test_frac, stratify=labels, random_state=SEED
        )
        rel_val = val_frac / (1.0 - test_frac)
        x_train, x_val, y_train, y_val = train_test_split(
            x_tv, y_tv, test_size=rel_val, stratify=y_tv, random_state=SEED
        )
    else:
        # rare-class safe split: shuffle, then for each class take 1 val + 1 test
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(paths))
        paths, labels = paths[order], labels[order]

        train_idx, val_idx, test_idx = [], [], []
        per_class_seen = Counter()
        for i, y in enumerate(labels):
            n = per_class_seen[y]
            if n == 0:
                val_idx.append(i)
            elif n == 1:
                test_idx.append(i)
            else:
                train_idx.append(i)
            per_class_seen[y] += 1
        x_train = paths[train_idx]; y_train = labels[train_idx]
        x_val   = paths[val_idx];   y_val   = labels[val_idx]
        x_test  = paths[test_idx];  y_test  = labels[test_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ───────────────────────────────────────────────────────────────
# tf.data pipeline
# ───────────────────────────────────────────────────────────────
def build_augment(strength: str = "medium"):
    """Build an augmentation pipeline. 'strong' is recommended for fine-tuning."""
    layers = [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08 if strength == "medium" else 0.15),
        tf.keras.layers.RandomZoom(0.10 if strength == "medium" else 0.18),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomBrightness(0.10 if strength == "medium" else 0.18),
        tf.keras.layers.RandomContrast(0.10 if strength == "medium" else 0.20),
    ]
    return tf.keras.Sequential(layers, name=f"augment_{strength}")


def _decode_resize(path, size):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [size, size])
    return img


def make_dataset(paths, labels, num_classes, *, training, img_size, batch, augment=None, oversample=False):
    if oversample and training:
        return _balanced_dataset(paths, labels, num_classes, img_size=img_size, batch=batch, augment=augment)

    paths_t = tf.constant(paths)
    labels_oh = tf.one_hot(labels, num_classes)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_oh))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 2048), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (_decode_resize(p, img_size), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch)
    if training and augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _balanced_dataset(paths, labels, num_classes, *, img_size, batch, augment):
    """Balanced minibatch sampler: every class is drawn with equal probability,
    so rare classes get repeated rather than swamped."""
    per_class = [tf.constant([p for p, l in zip(paths, labels) if l == c]) for c in range(num_classes)]
    weights = [1.0 / num_classes] * num_classes

    def make_class_ds(class_idx):
        cls_paths = per_class[class_idx]
        ds = tf.data.Dataset.from_tensor_slices(cls_paths).shuffle(
            buffer_size=tf.shape(cls_paths)[0], seed=SEED + class_idx, reshuffle_each_iteration=True
        ).repeat()
        oh = tf.one_hot(class_idx, num_classes)
        return ds.map(lambda p: (_decode_resize(p, img_size), oh),
                      num_parallel_calls=tf.data.AUTOTUNE)

    ds = tf.data.Dataset.sample_from_datasets(
        [make_class_ds(c) for c in range(num_classes)],
        weights=weights,
        seed=SEED,
    )
    ds = ds.batch(batch)
    if augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ───────────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────────
def build_model(num_classes: int, *, backbone_name: str, img_size: int):
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone_name!r}. Choices: {list(BACKBONES)}")
    spec = BACKBONES[backbone_name]

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    if spec["preprocess"] == "rescale_pm1":
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    else:
        x = inputs  # backbone handles its own preprocessing

    backbone = spec["ctor"]((img_size, img_size, 3))
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"mira_{backbone_name}"), backbone


# ───────────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-finetune", action="store_true", help="Skip the fine-tuning phase")
    ap.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    ap.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    ap.add_argument("--export-only", action="store_true", help="Skip training, export an existing model to TF.js")
    ap.add_argument("--backbone", choices=list(BACKBONES), default="mobilenet",
                    help="Feature extractor. 'efficientnet' usually scores ~3-5%% higher.")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE,
                    help="Input image edge in pixels. 224 default, try 260/300 for efficientnet.")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    ap.add_argument("--unfreeze", type=int, default=FINETUNE_UNFREEZE_LAYERS,
                    help="How many backbone layers to unfreeze in the fine-tune phase.")
    ap.add_argument("--oversample", action="store_true",
                    help="Balanced minibatch sampling — every class drawn equally. "
                         "Recommended when classes are very imbalanced (rare classes ≤10 imgs).")
    args = ap.parse_args()

    print(f"TensorFlow {tf.__version__}  ·  GPUs available: {tf.config.list_physical_devices('GPU')}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Backbone: {args.backbone}  ·  Image size: {args.img_size}  ·  Batch: {args.batch}")
    print(f"Oversample: {args.oversample}  ·  Label smoothing: {args.label_smoothing}")

    # ── Discover data ──────────────────────────────────────────
    paths, labels, class_names = discover_files(DATA_DIR)
    num_classes = len(class_names)
    print(f"Found {len(paths)} images across {num_classes} classes.")

    classes_json = {
        "index_to_label": class_names,
        "num_classes": num_classes,
        "image_size": args.img_size,
        "backbone": args.backbone,
    }
    (SAVE_DIR / "classes.json").write_text(json.dumps(classes_json, indent=2))

    # ── Splits ─────────────────────────────────────────────────
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = stratified_splits(paths, labels)
    print(f"Train: {len(x_tr)}  ·  Val: {len(x_val)}  ·  Test: {len(x_test)}")

    cw_arr = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_tr)
    class_weights = {i: float(w) for i, w in enumerate(cw_arr)}

    # ── Datasets ───────────────────────────────────────────────
    aug_medium = build_augment("medium")
    aug_strong = build_augment("strong")
    ds_train = make_dataset(
        x_tr, y_tr, num_classes,
        training=True, img_size=args.img_size, batch=args.batch,
        augment=aug_medium, oversample=args.oversample,
    )
    ds_val   = make_dataset(x_val, y_val, num_classes, training=False, img_size=args.img_size, batch=args.batch)
    ds_test  = make_dataset(x_test, y_test, num_classes, training=False, img_size=args.img_size, batch=args.batch)

    if args.export_only:
        print("Loading existing model for export-only run…")
        model = tf.keras.models.load_model(SAVE_DIR / "best.keras")
        export_to_tfjs(model)
        return

    # ── Build & train head ─────────────────────────────────────
    model, backbone = build_model(num_classes, backbone_name=args.backbone, img_size=args.img_size)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(HEAD_LR),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(SAVE_DIR / "best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, mode="max",
        restore_best_weights=True, verbose=1,
    )
    redlr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
    )

    # When oversampling, the dataset is infinite — must specify steps per epoch
    train_kwargs = {}
    if args.oversample:
        train_kwargs["steps_per_epoch"] = max(20, len(x_tr) // args.batch)

    print("\n──────── Phase 1: head only (frozen backbone) ────────")
    h1 = model.fit(
        ds_train, validation_data=ds_val, epochs=args.head_epochs,
        callbacks=[ckpt, early, redlr], class_weight=class_weights, verbose=2,
        **train_kwargs,
    )
    history = h1.history

    # ── Fine-tune the top of the backbone ──────────────────────
    if not args.no_finetune:
        print(f"\n──────── Phase 2: fine-tune top {args.unfreeze} layers of {args.backbone} ────────")
        backbone.trainable = True
        for layer in backbone.layers[:-args.unfreeze]:
            layer.trainable = False
        # Keep BatchNorm layers in inference mode — they were learned on ImageNet,
        # and small minibatches degrade their running stats.
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        # Stronger augmentation during fine-tune to combat overfitting
        ds_train_ft = make_dataset(
            x_tr, y_tr, num_classes,
            training=True, img_size=args.img_size, batch=args.batch,
            augment=aug_strong, oversample=args.oversample,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(FINETUNE_LR),
            loss=loss,
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")],
        )
        h2 = model.fit(
            ds_train_ft, validation_data=ds_val, epochs=args.finetune_epochs,
            callbacks=[ckpt, early, redlr], class_weight=class_weights, verbose=2,
            **train_kwargs,
        )
        for k, v in h2.history.items():
            history.setdefault(k, [])
            history[k].extend(v)

    # ── Evaluate on test ───────────────────────────────────────
    print("\n──────── Test set evaluation ────────")
    best = tf.keras.models.load_model(SAVE_DIR / "best.keras")
    test_metrics = best.evaluate(ds_test, return_dict=True, verbose=2)
    print("Test:", test_metrics)

    # Save history + test metrics
    (SAVE_DIR / "history.json").write_text(
        json.dumps(to_jsonable({"history": history, "test": test_metrics}), indent=2)
    )

    # ── Export to TF.js ────────────────────────────────────────
    export_to_tfjs(best)
    print("\nDone.  Files saved to:")
    print(f"  • {SAVE_DIR / 'best.keras'}")
    print(f"  • {SAVE_DIR / 'classes.json'}")
    print(f"  • {TFJS_DIR}/  (load this from the browser)")


# ───────────────────────────────────────────────────────────────
# TF.js export
# ───────────────────────────────────────────────────────────────
def export_to_tfjs(keras_model: tf.keras.Model):
    """Convert a Keras model to TensorFlow.js layers-model layout.

    Tries the official `tensorflowjs` package first; if its transitive
    imports (jax / flax / orbax / tf-hub) aren't installed — common on
    Windows where some of those have no wheel — falls back to a tiny
    pure-Python implementation that produces an equivalent
    layers-model directory the browser can load with
    `tf.loadLayersModel(...)`.
    """
    # Wipe the target dir for a clean export
    for f in TFJS_DIR.glob("*"):
        try:
            f.unlink()
        except IsADirectoryError:
            pass

    # ── Try official converter ─────────────────────────────────
    try:
        import tensorflowjs as tfjs
        print(f"Exporting via official tensorflowjs → {TFJS_DIR}")
        tfjs.converters.save_keras_model(keras_model, str(TFJS_DIR))
        _copy_classes_json()
        return
    except ImportError as e:
        print(f"[tfjs] official converter unavailable ({e.__class__.__name__}: {e})")
    except Exception as e:
        print(f"[tfjs] official converter raised {type(e).__name__}: {e}")

    # ── Minimal fallback ───────────────────────────────────────
    print(f"[tfjs] using built-in minimal converter → {TFJS_DIR}")
    _export_minimal(keras_model, TFJS_DIR)
    _copy_classes_json()
    print("[tfjs] export complete.")


def _export_minimal(keras_model: tf.keras.Model, output_dir: Path):
    """Pure-Python Keras → TF.js layers-model converter.

    Writes:
        output_dir/model.json              topology + weights manifest
        output_dir/group1-shard1of1.bin    concatenated float32 weights

    The browser loads this via `tf.loadLayersModel('.../model.json')`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Topology = the Keras model config as JSON.
    topology = json.loads(keras_model.to_json())

    # 2. Weights — every variable in the model (trainable and non-trainable,
    #    e.g. BatchNorm running mean/variance, Normalization mean/variance).
    weight_entries = []
    weight_chunks = []
    for var in keras_model.weights:
        name = var.name.split(":")[0]   # strip TF's ':0' suffix
        arr = var.numpy()
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        weight_entries.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": "float32",
        })
        weight_chunks.append(arr.tobytes(order="C"))

    # 3. Single-shard binary (float32, no quantisation — keeps things simple
    #    and lossless; ~10-20 MB for MobileNetV2 / EfficientNetB0).
    shard_name = "group1-shard1of1.bin"
    with open(output_dir / shard_name, "wb") as fh:
        for chunk in weight_chunks:
            fh.write(chunk)

    # 4. model.json — the manifest tf.loadLayersModel expects.
    model_json = {
        "format": "layers-model",
        "generatedBy": f"mira-minimal / TF {tf.__version__}",
        "convertedBy": "tensorflow.keras (mira)",
        "modelTopology": topology,
        "weightsManifest": [{
            "paths": [shard_name],
            "weights": weight_entries,
        }],
    }
    with open(output_dir / "model.json", "w") as fh:
        json.dump(model_json, fh)


def _copy_classes_json():
    src = SAVE_DIR / "classes.json"
    if src.exists():
        (TFJS_DIR / "classes.json").write_text(src.read_text())


if __name__ == "__main__":
    main()
