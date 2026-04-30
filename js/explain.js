/* =============================================================
   MIRA · Clinical Reading client
   - Compresses the source canvas (and DIP composite, if present) to JPEG
   - POSTs them along with the classifier's top-K to the local proxy
   - Returns Gemini's structured JSON for the caller to render
   ============================================================= */

const ENDPOINT = '/api/explain';
const JPEG_QUALITY = 0.85;
const MAX_DIM = 1024;

export async function requestReading({ sourceCanvas, compositeCanvas, topK }) {
  const sourceImage = canvasToJpegBase64(sourceCanvas);
  const compositeImage = compositeCanvas && hasPixels(compositeCanvas)
    ? canvasToJpegBase64(compositeCanvas)
    : null;

  const res = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sourceImage, compositeImage, topK: topK || [] }),
  });

  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(payload.error || `Server returned ${res.status}`);
  }
  return payload;
}

function hasPixels(canvas) {
  return canvas && canvas.width > 1 && canvas.height > 1;
}

function canvasToJpegBase64(canvas) {
  const max = Math.max(canvas.width, canvas.height);
  const target = max > MAX_DIM ? canvas : null;
  const surface = target ? downscale(canvas) : canvas;
  return surface.toDataURL('image/jpeg', JPEG_QUALITY).split(',', 2)[1];
}

function downscale(canvas) {
  const max = Math.max(canvas.width, canvas.height);
  const scale = MAX_DIM / max;
  const w = Math.max(1, Math.round(canvas.width * scale));
  const h = Math.max(1, Math.round(canvas.height * scale));
  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  tmp.getContext('2d').drawImage(canvas, 0, 0, w, h);
  return tmp;
}
