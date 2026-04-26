/* =============================================================
   MIRA · TensorFlow.js classifier wrapper
   Loads the MobileNetV2 transfer-learning model produced by
   model/train.py and exposes top-K inference for any canvas /
   image element.
   ============================================================= */

const DEFAULT_MODEL_URL = 'model/tfjs/model.json';
const DEFAULT_CLASSES_URL = 'model/tfjs/classes.json';

export class Classifier {
  constructor() {
    this.model = null;
    this.classes = [];
    this.imgSize = 224;
    this.ready = false;
    this.loadError = null;
  }

  /**
   * Load the TF.js model and class index map.
   * Resolves once both are in memory; rejects on network/parse errors.
   */
  async load(modelUrl = DEFAULT_MODEL_URL, classesUrl = DEFAULT_CLASSES_URL) {
    if (typeof tf === 'undefined') {
      throw new Error('TensorFlow.js is not loaded.');
    }

    // Class metadata
    const meta = await fetch(classesUrl).then((r) => {
      if (!r.ok) throw new Error(`classes.json: HTTP ${r.status}`);
      return r.json();
    });
    this.classes = meta.index_to_label || [];
    this.imgSize = meta.image_size || 224;

    // Layers model (saved via tfjs.converters.save_keras_model)
    this.model = await tf.loadLayersModel(modelUrl);

    // Warm-up run so the first real prediction isn't slow
    tf.tidy(() => {
      const dummy = tf.zeros([1, this.imgSize, this.imgSize, 3]);
      this.model.predict(dummy);
    });

    this.ready = true;
    return this;
  }

  /**
   * Run inference on a canvas / image / video element.
   * Returns { topK: [{ label, index, prob }], elapsedMs }.
   */
  predict(source, k = 5) {
    if (!this.ready) throw new Error('Classifier not loaded yet.');
    const t0 = performance.now();

    const probs = tf.tidy(() => {
      const img = tf.browser
        .fromPixels(source)
        .resizeBilinear([this.imgSize, this.imgSize])
        .toFloat()
        .expandDims(0);
      // Model's Rescaling layer handles -> [-1, 1]; pass raw 0-255 floats
      return this.model.predict(img).squeeze();
    });

    const data = probs.dataSync();
    probs.dispose();

    const idx = Array.from(data.keys()).sort((a, b) => data[b] - data[a]);
    const topK = idx.slice(0, k).map((i) => ({
      index: i,
      label: this.classes[i] || `Class ${i}`,
      prob: data[i],
    }));

    return { topK, elapsedMs: performance.now() - t0 };
  }
}

/** Pretty-print a class folder name like "0.0.Normal" → "Normal". */
export function prettyLabel(name) {
  if (!name) return '';
  // Strip leading "X." or "X.Y." prefix
  return name.replace(/^\d+(\.\d+)*\.\s*/, '').trim();
}
