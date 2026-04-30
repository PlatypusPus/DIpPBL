"""MIRA · Clinical Reading proxy.

Tiny Flask app that:
  - Serves the static frontend (index.html, js/, styles.css, model/)
  - Forwards explain requests to Google Gemini, keeping the API key server-side

Run:    python server/app.py
Config: copy .env.example to .env at the project root and fill in GEMINI_API_KEY
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
PORT = int(os.environ.get("PORT", "5174"))

if not API_KEY:
    sys.stderr.write(
        "[mira] GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.\n"
    )
    sys.exit(1)

import google.generativeai as genai
from PIL import Image

genai.configure(api_key=API_KEY)


PROMPT = """You are a clinical image-analysis assistant explaining retinal fundus diagnostics to medical students.

You will receive:
- A retinal fundus photograph (the source image).
- A diagnostic composite from MIRA's classical DIP pipeline. Vessels are tinted teal, lesion candidates amber, and the detected optic disc ringed in blue.
- The top predictions from a transfer-learned MobileNetV2 classifier trained on JSIEC-1000.

Reply with a single JSON object that matches this schema exactly. No prose outside the JSON. No markdown fences.

{
  "condition": {
    "name": "concise common name of the top-predicted condition",
    "summary": "one or two sentence definition aimed at a student"
  },
  "visualFindings": [
    {
      "feature": "a specific visual feature you actually see (e.g. cotton-wool spots, microaneurysms, hard exudates, drusen, neovascularisation, peripapillary atrophy, tessellation, large optic cup, etc.)",
      "location": "where in the image (quadrant or anatomical landmark)",
      "supports": "primary | differential | contradicts"
    }
  ],
  "pipelineCorroboration": "one or two sentences on whether the DIP composite's detected vessels, lesions and optic disc are spatially consistent with the top prediction",
  "confidence": {
    "level": "high | moderate | low",
    "note": "brief justification grounded in this image; mention the gap between top-1 and the differentials"
  },
  "disclaimer": "Educational analysis only - not a clinical diagnosis."
}

Be specific about features visible in this particular image. Do not invent findings. If image quality is poor, say so under confidence.
"""


app = Flask(__name__, static_folder=str(ROOT), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/api/explain", methods=["POST"])
def explain():
    data = request.get_json(silent=True) or {}
    source_b64 = data.get("sourceImage")
    composite_b64 = data.get("compositeImage")
    top_k = data.get("topK") or []

    if not source_b64:
        return jsonify({"error": "sourceImage is required"}), 400

    try:
        source_img = _decode_b64_image(source_b64)
    except Exception as exc:
        return jsonify({"error": f"could not decode sourceImage: {exc}"}), 400

    composite_img = None
    if composite_b64:
        try:
            composite_img = _decode_b64_image(composite_b64)
        except Exception:
            composite_img = None

    parts: list = [PROMPT]

    if top_k:
        lines = ["Classifier top predictions (highest first):"]
        for p in top_k[:5]:
            label = str(p.get("label", "?"))
            try:
                prob = float(p.get("prob", 0.0)) * 100
            except (TypeError, ValueError):
                prob = 0.0
            lines.append(f"- {label}: {prob:.1f}%")
        parts.append("\n".join(lines))

    parts.append("Source fundus image:")
    parts.append(source_img)
    if composite_img is not None:
        parts.append(
            "MIRA pipeline composite (vessels tinted teal, lesions amber, optic disc ringed blue):"
        )
        parts.append(composite_img)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            parts,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        )
    except Exception as exc:
        return jsonify({"error": f"Gemini call failed: {exc}"}), 502

    raw = (response.text or "").strip()
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return jsonify({"error": "Gemini returned non-JSON output", "raw": raw}), 502

    return jsonify(result)


def _decode_b64_image(b64: str) -> Image.Image:
    if isinstance(b64, str) and b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


if __name__ == "__main__":
    sys.stderr.write(
        f"[mira] serving http://localhost:{PORT}  (model={MODEL_NAME})\n"
    )
    app.run(host="127.0.0.1", port=PORT, debug=False)
