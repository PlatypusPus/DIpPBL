"""
MIRA — Offline prediction CLI.

Usage:
  python model/predict.py path/to/image.jpg
  python model/predict.py path/to/image.jpg --top 5

Loads the trained Keras model from model/saved/best.keras, runs a single
prediction, and prints the top-K class names with confidences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "model" / "saved"


def load_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return arr[None, ...]  # add batch dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr); sys.exit(1)

    model_path = SAVE_DIR / "best.keras"
    classes_path = SAVE_DIR / "classes.json"
    if not model_path.exists():
        print(f"Model not found: {model_path}.  Train first with: python model/train.py", file=sys.stderr); sys.exit(1)

    classes = json.loads(classes_path.read_text())
    label_of = classes["index_to_label"]
    size = classes.get("image_size", 224)

    model = tf.keras.models.load_model(model_path)
    x = load_image(args.image, size)
    probs = model.predict(x, verbose=0)[0]

    top_idx = np.argsort(probs)[::-1][: args.top]
    print(f"\n  {args.image.name}")
    print(f"  {'─' * 56}")
    for i, idx in enumerate(top_idx, 1):
        bar = "█" * int(probs[idx] * 30)
        print(f"  {i:>2}. {label_of[idx]:<40} {probs[idx]*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
