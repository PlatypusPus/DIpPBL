# MIRA — Medical Image Reasoning Atelier

A web-based **Digital Image Processing** teaching tool for retinal fundus
disease detection. It walks an uploaded image through 15 classical DIP stages
(channel extraction, CLAHE, Fourier spectrum, morphology, vessel segmentation,
Hough circles, …) and runs an in-browser **MobileNetV2 / EfficientNet**
classifier trained on [JSIEC-1000](https://www.kaggle.com/datasets/linchundan/fundusimage1000)
(39 disease classes).

Pure HTML/CSS/JS frontend (OpenCV.js + TensorFlow.js, no build step). Python is
only needed to train the model or run the optional Gemini explanation proxy.

## Quick start

```powershell
python -m http.server 8000   # ES modules need an HTTP origin, not file://
# open http://localhost:8000
```

Upload an image → **Run Pipeline** to step through the DIP stages. If
`model/tfjs/` is present, the inference panel shows the top-5 predictions.

For the **Clinical Reading** (Gemini) panel, serve through the Python proxy
instead — see [Clinical Reading](#clinical-reading-gemini) below.

## Train the classifier (~20–40 min on CPU)

Requires **Python 3.10 or 3.11** (TensorFlow has no Windows wheels for 3.13+).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --prefer-binary -r model\requirements.txt
python model\train.py          # writes model/saved/ and model/tfjs/
```

Higher accuracy (~+5%): `python model\train.py --backbone efficientnet --img-size 260 --head-epochs 20 --finetune-epochs 25 --unfreeze 60 --oversample`

CLI single-image test: `python model\predict.py <path-to-image.JPG>`

## Clinical Reading (Gemini)

A tiny Flask proxy ([server/app.py](server/app.py)) serves the frontend and a
`/api/explain` endpoint that asks Gemini to read the fundus image plus the
pipeline composite and return an image-grounded explanation. The API key stays
server-side. *Educational second opinion, not a clinical tool.*

```powershell
pip install -r server\requirements.txt
copy .env.example .env          # then set GEMINI_API_KEY (https://aistudio.google.com/app/apikey)
python server\app.py            # serves http://localhost:5174
```

In-app: upload → **Run Pipeline** → **Generate Reading**.

Env vars (`.env`): `GEMINI_API_KEY` (required), `GEMINI_MODEL`
(default `gemini-2.5-flash`), `PORT` (default `5174`).

## Deploy (Render)

The repo includes [render.yaml](render.yaml). Render dashboard →
**New → Blueprint** → pick this repo → set `GEMINI_API_KEY` in the Environment
tab → deploy. The Flask proxy serves both the frontend and `/api/explain`.

> Free tier sleeps after 15 min idle, so the first request after a nap takes ~30s.

## DIP stages

| #  | Stage                   | Default            | Other methods                 |
|----|-------------------------|--------------------|-------------------------------|
| 01 | Image Acquisition       | (raw RGB)          | —                             |
| 02 | Channel Extraction      | Green              | Red, Blue, Y, HSV-V, Lab-L    |
| 03 | Illumination Correction | Mean Subtract      | Median, Morphological, None   |
| 04 | Noise Reduction         | Gaussian           | Median, Bilateral, None       |
| 05 | Contrast Enhancement    | CLAHE              | Histogram Eq, Gamma, None     |
| 06 | Frequency Spectrum      | Log Magnitude DFT  | Phase                         |
| 07 | Image Sharpening        | Unsharp Mask       | Laplacian, High-Boost, None   |
| 08 | Morphological Op        | Black-Hat          | Top-Hat, Opening, Closing     |
| 09 | Edge Detection          | Canny              | Sobel, Laplacian, Prewitt     |
| 10 | Vessel Segmentation     | Otsu               | Adaptive Mean/Gaussian, Fixed |
| 11 | Region Cleanup          | Area Filter        | Closing+Area, None            |
| 12 | Skeletonization         | Lantuéjoul         | Distance Ridge, None          |
| 13 | Lesion Detection        | Bright (Top-Hat)   | Dark (Black-Hat), None        |
| 14 | Optic Disc Localisation | Hough Circle       | Brightest Centroid, None      |
| 15 | Diagnostic Composite    | Layered Overlay    | toggle vessels/lesions/OD     |

Click any stage pill to swap methods — the pipeline re-runs from that stage down.

## Viewer controls

**Wheel** zoom · **drag** pan · **double-click** reset · `+` `−` `0` keys.
Pan/zoom is synced across both panels.

## Tech stack

OpenCV.js 4.x (DIP) · TensorFlow.js 4.x (inference) · TensorFlow 2.15 + Keras
(training, transfer learning) · pure ES modules, no bundler.

## Troubleshooting

- **"Failed to fetch model.json"** — you opened `file://`. Use `http://localhost:8000`.
- **"Model not found"** — train it (`python model\train.py`) or copy a teammate's `model/tfjs/`.
- **`pip install` fails on tensorflow** — Python too new; use 3.10 or 3.11.
