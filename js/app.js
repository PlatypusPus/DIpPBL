/* =============================================================
   MIRA · App orchestration
   - Wires UI to the DIP pipeline
   - Handles upload, run, stage selection, method swapping
   - Renders histograms and the gallery strip
   ============================================================= */

import { STAGES, Pipeline } from './pipeline.js';
import { Classifier, prettyLabel } from './classifier.js';
import { SyncedViewer } from './viewer.js';

// Stages that appear in the bottom Learning Bar (per design blueprint)
const LEARNING_BAR_IDS = ['channel', 'contrast', 'segmentation', 'lesion'];

// ─── DOM refs ──────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const ui = {
  fileInput: $('file-input'),
  runBtn: $('run-btn'),
  runBtnLabel: $('run-btn-label'),
  sourceCanvas: $('source-canvas'),
  sourceEmpty: $('source-empty'),
  sourceMeta: $('source-meta'),
  outputCanvas: $('output-canvas'),
  outputEmpty: $('output-empty'),
  outputTitle: $('output-title'),
  toggleVessels: $('toggle-vessels'),
  toggleLesions: $('toggle-lesions'),
  toggleOptic: $('toggle-optic'),
  detailMethods: $('detail-methods'),
  railList: $('rail-list'),
  stageProgress: $('stage-progress'),
  opencvStatus: $('opencv-status'),
  learningBar: $('learning-bar'),
  detailEyebrow: $('detail-eyebrow'),
  detailTitle: $('detail-title'),
  detailTags: $('detail-tags'),
  stageCanvas: $('stage-canvas'),
  stageCanvasEmpty: $('stage-canvas-empty'),
  histCanvas: $('hist-canvas'),
  statMean: $('stat-mean'),
  statStd: $('stat-std'),
  statRange: $('stat-range'),
  statEntropy: $('stat-entropy'),
  rationaleText: $('rationale-text'),
  rationaleFormula: $('rationale-formula'),
  rationaleParams: $('rationale-params'),
  galleryStrip: $('gallery-strip'),
  toast: $('toast'),
  // ─── Inference panel ─────────────────────────────────────────
  inferencePanel: $('inference-panel'),
  inferenceStatus: $('inference-status'),
  primaryName: $('primary-name'),
  primaryClass: $('primary-class'),
  primaryBar: $('primary-bar'),
  primaryConf: $('primary-conf'),
  primaryTime: $('primary-time'),
  primaryNclasses: $('primary-nclasses'),
  inferenceList: $('inference-list'),
};

// ─── State ─────────────────────────────────────────────────────
let cv = null;
let pipeline = null;
let activeStageId = 'acquisition';
let hasRun = false;
const classifier = new Classifier();
let classifierLoading = false;
let viewer = null;

// ─── Bootstrap ─────────────────────────────────────────────────
window.addEventListener('opencv-ready', () => {
  cv = window.cv;
  pipeline = new Pipeline(cv);
  setOpenCvStatus('ok', 'OpenCV runtime ready · ' + cv.getBuildInformation().split('\n')[0]);
  init();
});

// Fallback if cv is already available (e.g. cached)
if (typeof window !== 'undefined' && window.cv && window.cv.getBuildInformation) {
  cv = window.cv;
  pipeline = new Pipeline(cv);
  setOpenCvStatus('ok', 'OpenCV runtime ready');
  init();
}

function setOpenCvStatus(kind, msg) {
  ui.opencvStatus.innerHTML =
    `<span class="status-dot status-dot--${kind}"></span><span>${msg}</span>`;
}

function init() {
  buildRail();
  buildLearningBar();
  buildGallery();
  renderActiveStage(); // shows empty state
  attachEvents();
  initViewer();
  loadClassifierIfPresent();
}

function initViewer() {
  const sourceBody = document.querySelector('.viewer-pane--source .pane-body');
  const outputBody = document.querySelector('.viewer-pane--output .pane-body');
  const zoomLevel = document.getElementById('zoom-level');
  viewer = new SyncedViewer(
    [
      { container: sourceBody, canvas: ui.sourceCanvas },
      { container: outputBody, canvas: ui.outputCanvas },
    ],
    { onChange: (s) => { zoomLevel.textContent = `${Math.round(s * 100)}%`; } },
  );
  document.querySelectorAll('.viewer-zoom [data-zoom]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const a = btn.dataset.zoom;
      if (a === 'in') viewer.zoomIn();
      else if (a === 'out') viewer.zoomOut();
      else if (a === 'reset') viewer.reset();
    });
  });
}

// ─── Classifier load ───────────────────────────────────────────
async function loadClassifierIfPresent() {
  if (typeof tf === 'undefined') {
    setInferenceStatus('err', 'TensorFlow.js failed to load');
    return;
  }
  // Probe for the model so we don't 404 noisily on a fresh checkout
  try {
    const probe = await fetch('model/tfjs/model.json', { method: 'HEAD' });
    if (!probe.ok) {
      setInferenceStatus('load', 'Model not found · run model/train.py');
      ui.primaryName.textContent = 'No trained model';
      ui.primaryClass.textContent = 'see model/train.py to build one';
      return;
    }
  } catch {
    setInferenceStatus('load', 'Model not found · run model/train.py');
    ui.primaryName.textContent = 'No trained model';
    ui.primaryClass.textContent = 'see model/train.py to build one';
    return;
  }

  classifierLoading = true;
  setInferenceStatus('load', 'Loading classifier weights…');
  try {
    await classifier.load();
    setInferenceStatus('ok', `Ready · ${classifier.classes.length} classes · ${classifier.imgSize}×${classifier.imgSize}`);
    ui.primaryNclasses.textContent = String(classifier.classes.length);
    ui.primaryName.textContent = 'Awaiting image';
    ui.primaryClass.textContent = 'upload a fundus image and run the pipeline';
    // If the user already uploaded an image, run inference now
    if (pipeline && pipeline.hasSource()) runInference();
  } catch (err) {
    console.error(err);
    setInferenceStatus('err', 'Model load failed · see console');
  } finally {
    classifierLoading = false;
  }
}

function setInferenceStatus(kind, msg) {
  ui.inferenceStatus.innerHTML =
    `<span class="status-dot status-dot--${kind}"></span><span>${msg}</span>`;
}

// ─── Inference ─────────────────────────────────────────────────
function runInference() {
  if (!classifier.ready) return;
  if (!pipeline.hasSource()) return;
  try {
    const { topK, elapsedMs } = classifier.predict(ui.sourceCanvas, 5);
    renderInference(topK, elapsedMs);
  } catch (err) {
    console.error(err);
    setInferenceStatus('err', 'Inference failed · ' + err.message);
  }
}

function renderInference(topK, elapsedMs) {
  if (!topK || !topK.length) return;
  const top = topK[0];
  ui.primaryName.textContent = prettyLabel(top.label);
  ui.primaryClass.textContent = top.label;
  ui.primaryConf.textContent = (top.prob * 100).toFixed(1) + '%';
  ui.primaryBar.style.width = (top.prob * 100).toFixed(1) + '%';
  ui.primaryTime.textContent = `${elapsedMs.toFixed(0)} ms`;

  ui.inferenceList.innerHTML = '';
  topK.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'inf-row' + (i === 0 ? ' is-top' : '');
    row.innerHTML = `
      <span class="inf-rank">${String(i + 1).padStart(2, '0')}</span>
      <div class="inf-name"><strong>${prettyLabel(p.label)}</strong><span>${p.label}</span></div>
      <span class="inf-pct">${(p.prob * 100).toFixed(1)}%</span>
      <div class="inf-bar"><div class="inf-bar-fill" style="width:${(p.prob * 100).toFixed(1)}%"></div></div>
    `;
    ui.inferenceList.appendChild(row);
  });
}

// ─── Rail ──────────────────────────────────────────────────────
function buildRail() {
  ui.railList.innerHTML = '';
  for (const stage of STAGES) {
    const li = document.createElement('li');
    li.className = 'rail-item';
    li.dataset.stage = stage.id;
    li.innerHTML = `
      <span class="rail-num">${stage.num}</span>
      <div>
        <div class="rail-name">${stage.name}</div>
        <div class="rail-sub">${stage.sub}</div>
      </div>
      <span class="rail-status-tag tag-pending">Pending</span>
    `;
    li.addEventListener('click', () => setActiveStage(stage.id));
    ui.railList.appendChild(li);
  }
  refreshRailHighlights();
}

function refreshRailHighlights() {
  let done = 0;
  ui.railList.querySelectorAll('.rail-item').forEach((li) => {
    const sid = li.dataset.stage;
    li.classList.toggle('is-active', sid === activeStageId);

    const tag = li.querySelector('.rail-status-tag');
    const stage = pipeline.getStage(sid);
    const ran = !!pipeline.state[sid];
    li.classList.toggle('is-done', ran);
    if (ran) done++;

    tag.className = 'rail-status-tag';
    if (!ran) {
      tag.classList.add('tag-pending');
      tag.textContent = 'Pending';
    } else if (stage.fixed) {
      tag.classList.add('tag-optimal');
      tag.textContent = 'Ready';
    } else if (pipeline.isOptimal(sid)) {
      tag.classList.add('tag-optimal');
      tag.textContent = 'Optimal';
    } else {
      tag.classList.add('tag-modified');
      tag.textContent = 'Modified';
    }
  });
  ui.stageProgress.textContent = `${done} / ${STAGES.length}`;
}

// ─── Learning bar ──────────────────────────────────────────────
function buildLearningBar() {
  ui.learningBar.innerHTML = '';
  LEARNING_BAR_IDS.forEach((sid, idx) => {
    const stage = pipeline.getStage(sid);
    const card = document.createElement('div');
    card.className = 'lb-stage';
    card.dataset.stage = sid;
    card.innerHTML = `
      <div class="lb-stage-head">
        <div>
          <div class="lb-stage-num">STAGE ${String(idx + 1).padStart(2, '0')}</div>
          <div class="lb-stage-name">${stage.name}</div>
        </div>
        <span class="rail-status-tag tag-optimal" data-tag>Optimal</span>
      </div>
      <div class="lb-pills" data-pills></div>
    `;

    const pills = card.querySelector('[data-pills]');
    for (const m of stage.methods) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'lb-pill';
      b.dataset.method = m.id;
      b.innerHTML = (m.optimal ? '<span class="pill-star"></span>' : '') + m.name;
      b.addEventListener('click', () => onMethodPick(sid, m.id));
      pills.appendChild(b);
    }
    card.addEventListener('click', (e) => {
      if (e.target.closest('.lb-pill')) return;
      setActiveStage(sid);
    });
    ui.learningBar.appendChild(card);
  });
  refreshLearningBar();
}

function refreshLearningBar() {
  ui.learningBar.querySelectorAll('.lb-stage').forEach((card) => {
    const sid = card.dataset.stage;
    card.classList.toggle('is-active', sid === activeStageId);

    // Tag
    const tag = card.querySelector('[data-tag]');
    if (pipeline.isOptimal(sid)) {
      tag.className = 'rail-status-tag tag-optimal';
      tag.textContent = 'Optimal';
    } else {
      tag.className = 'rail-status-tag tag-modified';
      tag.textContent = 'Modified';
    }

    // Pills
    const selected = pipeline.config[sid];
    card.querySelectorAll('.lb-pill').forEach((p) => {
      p.classList.toggle('is-selected', p.dataset.method === selected);
    });
  });
}

// ─── Gallery ───────────────────────────────────────────────────
function buildGallery() {
  ui.galleryStrip.innerHTML = '';
  for (const stage of STAGES) {
    const div = document.createElement('div');
    div.className = 'thumb';
    div.dataset.stage = stage.id;
    div.innerHTML = `
      <div class="thumb-canvas-wrap is-empty">
        <canvas></canvas>
      </div>
      <div class="thumb-meta">
        <div class="thumb-num">${stage.num} · ${stage.eyebrow.split('·').slice(-1)[0].trim()}</div>
        <div class="thumb-name">${stage.name}</div>
      </div>
    `;
    div.addEventListener('click', () => setActiveStage(stage.id));
    ui.galleryStrip.appendChild(div);
  }
}

function refreshGallery() {
  ui.galleryStrip.querySelectorAll('.thumb').forEach((thumb) => {
    const sid = thumb.dataset.stage;
    thumb.classList.toggle('is-active', sid === activeStageId);

    const wrap = thumb.querySelector('.thumb-canvas-wrap');
    const canvas = wrap.querySelector('canvas');
    if (pipeline.state[sid]) {
      wrap.classList.remove('is-empty');
      pipeline.renderStage(sid, canvas);
    } else {
      wrap.classList.add('is-empty');
      // Clear canvas
      const ctx = canvas.getContext('2d');
      canvas.width = 1; canvas.height = 1;
      ctx.clearRect(0, 0, 1, 1);
    }
  });
}

// ─── Active stage rendering ────────────────────────────────────
function setActiveStage(stageId) {
  activeStageId = stageId;
  refreshRailHighlights();
  refreshLearningBar();
  refreshGallery();
  renderActiveStage();
}

function renderActiveStage() {
  const stage = pipeline.getStage(activeStageId);
  if (!stage) return;

  ui.detailEyebrow.textContent = stage.eyebrow;
  ui.detailTitle.textContent = stage.detailTitle;

  // Tags row
  ui.detailTags.innerHTML = '';
  if (pipeline.state[stage.id]) {
    if (stage.fixed) {
      ui.detailTags.innerHTML = `<span class="rail-status-tag tag-optimal">Ready</span>`;
    } else if (pipeline.isOptimal(stage.id)) {
      ui.detailTags.innerHTML = `<span class="rail-status-tag tag-optimal">Optimal Method</span>`;
    } else {
      ui.detailTags.innerHTML = `<span class="rail-status-tag tag-modified">Modified Method</span>`;
    }
    if (stage.branch) {
      ui.detailTags.innerHTML +=
        ` <span class="rail-status-tag tag-modified" title="Visualisation branch — does not feed downstream stages">Branch</span>`;
    }
  } else {
    ui.detailTags.innerHTML = `<span class="rail-status-tag tag-pending">Awaiting Run</span>`;
  }

  // Method pills (rendered for any stage with selectable methods)
  renderDetailMethods(stage);

  // Stage canvas
  const ran = !!pipeline.state[stage.id];
  ui.stageCanvas.classList.toggle('is-loaded', ran);
  ui.stageCanvasEmpty.style.display = ran ? 'none' : '';
  if (ran) pipeline.renderStage(stage.id, ui.stageCanvas);

  // Histogram + stats
  if (ran) {
    drawHistogram(pipeline.statsFor(stage.id));
  } else {
    clearHistogram();
  }

  // Rationale
  if (stage.fixed && stage.description) {
    ui.rationaleText.textContent = stage.description;
    ui.rationaleFormula.textContent = '';
    ui.rationaleParams.innerHTML = '';
  } else {
    const m = pipeline.getMethod(stage.id);
    if (m) {
      ui.rationaleText.textContent = m.rationale || '';
      ui.rationaleFormula.textContent = m.formula || '';
      ui.rationaleParams.innerHTML = renderParamChips(m.params);
    } else {
      ui.rationaleText.textContent = 'Pick a method pill to see details.';
      ui.rationaleFormula.textContent = '';
      ui.rationaleParams.innerHTML = '';
    }
  }
}

function renderDetailMethods(stage) {
  ui.detailMethods.innerHTML = '';
  if (!stage.methods || stage.methods.length <= 1) return;

  const label = document.createElement('span');
  label.className = 'detail-methods-label';
  label.textContent = 'Method';
  ui.detailMethods.appendChild(label);

  const selected = pipeline.config[stage.id];
  for (const m of stage.methods) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'lb-pill' + (m.id === selected ? ' is-selected' : '');
    b.dataset.method = m.id;
    b.innerHTML = (m.optimal ? '<span class="pill-star"></span>' : '') + m.name;
    b.addEventListener('click', () => onMethodPick(stage.id, m.id));
    ui.detailMethods.appendChild(b);
  }
}

function renderParamChips(params) {
  if (!params) return '';
  const entries = Object.entries(params);
  if (!entries.length) return '';
  return entries
    .map(([k, v]) => `<span class="param-chip">${k}<strong>${formatVal(v)}</strong></span>`)
    .join('');
}
function formatVal(v) {
  if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(2);
  return String(v);
}

// ─── Histogram drawing ─────────────────────────────────────────
function drawHistogram(stats) {
  if (!stats) return clearHistogram();
  const c = ui.histCanvas;
  const dpr = window.devicePixelRatio || 1;
  const cssW = c.clientWidth;
  const cssH = c.clientHeight;
  c.width = Math.floor(cssW * dpr);
  c.height = Math.floor(cssH * dpr);
  const g = c.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, cssW, cssH);

  // Bg grid
  g.strokeStyle = 'rgba(191, 201, 196, 0.06)';
  g.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (cssH / 4) * i;
    g.beginPath();
    g.moveTo(0, y); g.lineTo(cssW, y);
    g.stroke();
  }

  const max = Math.max(...stats.bins);
  if (max === 0) return;
  const barW = cssW / 256;

  // Bars
  const grad = g.createLinearGradient(0, 0, 0, cssH);
  grad.addColorStop(0, 'rgba(148, 211, 193, 0.95)');
  grad.addColorStop(1, 'rgba(148, 211, 193, 0.15)');
  g.fillStyle = grad;
  for (let i = 0; i < 256; i++) {
    const h = (stats.bins[i] / max) * (cssH - 4);
    g.fillRect(i * barW, cssH - h, Math.max(barW - 0.4, 0.6), h);
  }

  // Mean marker
  const mx = (stats.mean / 255) * cssW;
  g.strokeStyle = 'rgba(241, 194, 125, 0.85)';
  g.setLineDash([3, 3]);
  g.beginPath();
  g.moveTo(mx, 0); g.lineTo(mx, cssH);
  g.stroke();
  g.setLineDash([]);

  // Stats
  ui.statMean.textContent = stats.mean.toFixed(1);
  ui.statStd.textContent = stats.std.toFixed(1);
  ui.statRange.textContent = `${stats.min} – ${stats.max}`;
  ui.statEntropy.textContent = stats.entropy.toFixed(2);
}

function clearHistogram() {
  const c = ui.histCanvas;
  const g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  ui.statMean.textContent = '—';
  ui.statStd.textContent = '—';
  ui.statRange.textContent = '—';
  ui.statEntropy.textContent = '—';
}

// ─── Events ────────────────────────────────────────────────────
function attachEvents() {
  ui.fileInput.addEventListener('change', onFileSelected);
  ui.runBtn.addEventListener('click', () => runPipeline());

  ui.toggleVessels.addEventListener('change', () => {
    pipeline.showVessels = ui.toggleVessels.checked;
    if (hasRun) {
      pipeline.runComposite();
      paintOutput();
      refreshGallery();
    }
  });
  ui.toggleLesions.addEventListener('change', () => {
    pipeline.showLesions = ui.toggleLesions.checked;
    if (hasRun) {
      pipeline.runComposite();
      paintOutput();
      refreshGallery();
    }
  });
  ui.toggleOptic.addEventListener('change', () => {
    pipeline.showOpticDisc = ui.toggleOptic.checked;
    if (hasRun) {
      pipeline.runComposite();
      paintOutput();
      refreshGallery();
    }
  });
}

async function onFileSelected(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const img = await loadImage(file);
    paintImageToCanvas(img, ui.sourceCanvas);
    ui.sourceCanvas.classList.add('is-loaded');
    ui.sourceEmpty.style.display = 'none';

    const meta =
      `<span>RES ${img.naturalWidth}×${img.naturalHeight}</span>` +
      `<span>CH RGBA</span>` +
      `<span>${(file.size / 1024).toFixed(0)} KB</span>`;
    ui.sourceMeta.innerHTML = meta;

    if (pipeline) {
      pipeline.setSourceFromCanvas(ui.sourceCanvas);
      ui.runBtn.disabled = false;
      ui.runBtnLabel.textContent = 'Run Pipeline';
      toast('Source loaded · ready to run', 'ok');
      if (viewer) viewer.reset();
      // Show source in detail if user is on stage 1
      if (activeStageId === 'acquisition') renderActiveStage();
      refreshRailHighlights();
      refreshGallery();
    }
    // Fire ML inference on upload — independent of the DIP pipeline
    if (classifier.ready) runInference();
  } catch (err) {
    console.error(err);
    toast('Could not load image', 'error');
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { resolve(img); URL.revokeObjectURL(url); };
    img.onerror = (e) => { reject(e); URL.revokeObjectURL(url); };
    img.src = url;
  });
}

function paintImageToCanvas(img, canvas) {
  // Cap size for performance, preserve aspect
  const MAX = 1200;
  const scale = Math.min(1, MAX / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
}

function paintOutput() {
  if (pipeline.state.composite) {
    ui.outputCanvas.classList.add('is-loaded');
    ui.outputEmpty.style.display = 'none';
    pipeline.renderStage('composite', ui.outputCanvas);
    ui.outputTitle.textContent = 'Diagnostic Composite';
  }
}

async function runPipeline() {
  if (!pipeline.hasSource()) {
    toast('Upload a source image first', 'error');
    return;
  }
  ui.runBtn.disabled = true;
  ui.runBtnLabel.textContent = 'Processing…';
  // micro-yield so UI updates before WASM heavy lifting
  await new Promise((r) => requestAnimationFrame(r));

  const t0 = performance.now();
  try {
    pipeline.runAll();
    paintOutput();
    refreshGallery();
    refreshRailHighlights();
    refreshLearningBar();
    renderActiveStage();
    if (classifier.ready) runInference();
    hasRun = true;
    const ms = Math.round(performance.now() - t0);
    toast(`Pipeline complete · ${ms} ms`, 'ok');
    ui.runBtnLabel.textContent = 'Re-run Pipeline';
  } catch (err) {
    console.error(err);
    toast('Pipeline failed: ' + err.message, 'error');
    ui.runBtnLabel.textContent = 'Run Pipeline';
  } finally {
    ui.runBtn.disabled = false;
  }
}

function onMethodPick(stageId, methodId) {
  pipeline.setMethod(stageId, methodId);
  // Highlight changes immediately, even if we haven't run yet
  refreshLearningBar();
  if (activeStageId === stageId) renderActiveStage();
  refreshRailHighlights();

  if (hasRun) {
    // Re-run from this stage onward
    try {
      pipeline.runFrom(stageId);
      pipeline.runComposite();
      paintOutput();
      refreshGallery();
      renderActiveStage();
      ui.runBtnLabel.textContent = 'Re-run Pipeline';
      toast(`Updated ${pipeline.getStage(stageId).name}`, 'ok');
    } catch (err) {
      console.error(err);
      toast('Re-run failed: ' + err.message, 'error');
    }
  } else if (pipeline.hasSource()) {
    ui.runBtnLabel.textContent = 'Run Pipeline';
  }
}

// ─── Toast ─────────────────────────────────────────────────────
let toastTimer;
function toast(msg, kind = 'ok') {
  ui.toast.textContent = msg;
  ui.toast.className = 'toast is-visible' + (kind ? ' is-' + kind : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    ui.toast.className = 'toast' + (kind ? ' is-' + kind : '');
  }, 2400);
}
