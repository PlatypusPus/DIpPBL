# MIRA — Medical Image Reasoning Atelier

A web-based teaching tool for **Digital Image Processing** built around retinal
fundus disease detection. Walks through 15 classical DIP stages (channel
extraction, illumination correction, CLAHE, Fourier spectrum, morphology,
thresholding, skeletonization, Hough circles, …) on the image you upload, and
runs an in-browser **MobileNetV2 / EfficientNet** classifier trained on the
[JSIEC-1000](https://www.kaggle.com/datasets/linchundan/fundusimage1000)
dataset (39 disease classes).

Built for a college DIP project. Pure HTML / CSS / JS frontend (OpenCV.js +
TensorFlow.js); Python is only needed to train the model.

---

## Quick start (frontend only — no training)

If a teammate has already shared a pre-trained model, drop their `model/tfjs/`
folder into yours and you can skip straight to step 3.

```powershell
# 1. Clone
git clone <repo-url> mira
cd mira

# 2. (optional) Get the dataset — JSIEC-1000 from Kaggle
#    https://www.kaggle.com/datasets/linchundan/fundusimage1000
#    Extract so the structure looks like:  samples/1000images/<class>/<image>.jpg
#    (samples/ is git-ignored)

# 3. Serve the project — ES modules need an HTTP origin, not file://
python -m http.server 8000
# then open http://localhost:8000
```

Upload any image, click **Run Pipeline** to walk through the DIP stages, and
the inference panel will show the top-5 predictions if a model is present.

---

## Train the classifier (one-time, ~20–40 min on CPU)

Requires **Python 3.10 or 3.11** on Windows. TensorFlow has no wheels for
Python 3.13/3.14 yet on Windows, so don't use those.

```powershell
# 1. Make a venv with Python 3.10 or 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip + install deps (skip tensorflowjs — we use a built-in
#    minimal converter; tensorflowjs's transitive deps don't install
#    cleanly on Windows)
python -m pip install --upgrade pip
pip install --prefer-binary -r model\requirements.txt

# 3. Train + export to TF.js format (writes model/saved/ and model/tfjs/)
python model\train.py

# Optional flags for higher accuracy (~+5%):
#   python model\train.py --backbone efficientnet --img-size 260 \
#                         --head-epochs 20 --finetune-epochs 25 \
#                         --unfreeze 60 --oversample
```

After training, refresh the browser — the inference panel will pick up
`model/tfjs/` automatically.

### Test a single image from the CLI

```powershell
python model\predict.py samples\1000images\0.0.Normal\some-image.JPG
```

---

## Project structure

```
mira/
├── index.html              # App shell
├── styles.css              # Clinical Atelier theme
├── js/
│   ├── app.js              # UI orchestration
│   ├── pipeline.js         # 15-stage DIP pipeline (OpenCV.js)
│   ├── classifier.js       # TensorFlow.js inference
│   └── viewer.js           # Synced pan/zoom for both panels
├── model/
│   ├── train.py            # Training pipeline (transfer learning)
│   ├── predict.py          # CLI single-image inference
│   ├── requirements.txt    # Python deps (TF 2.15, sklearn, pillow, …)
│   ├── saved/              # ⛔ gitignored — Keras model + class map
│   └── tfjs/               # ⛔ gitignored — browser-loadable model
├── samples/                # ⛔ gitignored — get from Kaggle (see above)
└── mira_project_documentation_design.md   # Original design blueprint
```

---

## DIP stages (in pipeline order)

| #  | Stage                  | Default method     | Other methods                      |
|----|------------------------|--------------------|------------------------------------|
| 01 | Image Acquisition      | (raw RGB)          | —                                  |
| 02 | Channel Extraction     | Green              | Red, Blue, Y, HSV-V, Lab-L         |
| 03 | Illumination Correction| Mean Subtract      | Median, Morphological, None        |
| 04 | Noise Reduction        | Gaussian           | Median, Bilateral, None            |
| 05 | Contrast Enhancement   | CLAHE              | Histogram Eq, Gamma, None          |
| 06 | Frequency Spectrum     | Log Magnitude DFT  | Phase                              |
| 07 | Image Sharpening       | Unsharp Mask       | Laplacian, High-Boost, None        |
| 08 | Morphological Op       | Black-Hat          | Top-Hat, Opening, Closing          |
| 09 | Edge Detection         | Canny              | Sobel, Laplacian, Prewitt          |
| 10 | Vessel Segmentation    | Otsu               | Adaptive Mean/Gaussian, Fixed      |
| 11 | Region Cleanup         | Area Filter        | Closing+Area, None                 |
| 12 | Skeletonization        | Lantuéjoul         | Distance Ridge, None               |
| 13 | Lesion Detection       | Bright (Top-Hat)   | Dark (Black-Hat), None             |
| 14 | Optic Disc Localisation| Hough Circle       | Brightest Centroid, None           |
| 15 | Diagnostic Composite   | Layered Overlay    | (toggleable: vessels/lesions/OD)   |

Click any pill to swap methods — the rationale, formula, and parameters update
in the detail panel; the pipeline re-runs from that stage downstream.

---

## Viewer controls

- **Wheel** — zoom toward cursor (synced across both panels)
- **Click + drag** — pan
- **Double-click** — reset
- **Keyboard** — `+` `−` `0`
- **Bottom toolbar** — buttons + zoom level

---

## Tech stack

- **OpenCV.js 4.x** — all DIP operations
- **TensorFlow.js 4.x** — in-browser inference
- **TensorFlow 2.15 + Keras** — training (transfer learning from
  MobileNetV2 / EfficientNet)
- Pure ES modules — no bundler, no build step

---

## Troubleshooting

**"Failed to fetch model.json"** — you opened `file://`. Run
`python -m http.server 8000` and use `http://localhost:8000`.

**Inference panel says "Model not found"** — train the model
(`python model\train.py`) or get the `model/tfjs/` folder from a teammate.

**`pip install` fails on tensorflow** — your Python version is too new.
Use Python 3.10 or 3.11.
