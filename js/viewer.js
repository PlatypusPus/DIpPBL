/* =============================================================
   MIRA · Synced pan/zoom controller for the split viewer.
   - Wheel zooms toward the cursor
   - Left-mouse / touch drag pans
   - Both panels share a single transform so they stay locked
   - Keyboard:  +  −  to zoom,  0  to reset
   ============================================================= */

export class SyncedViewer {
  constructor(panes, opts = {}) {
    this.panes = panes;            // [{ container, canvas }, …]
    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.minScale = opts.minScale ?? 1;
    this.maxScale = opts.maxScale ?? 12;
    this.zoomStep = opts.zoomStep ?? 1.25;
    this.onChange = opts.onChange ?? (() => {});
    this._attach();
    this._apply();
  }

  // ─── Public API ──────────────────────────────────────────────
  zoomAt(targetScale, cx = 0, cy = 0) {
    const newScale = clamp(targetScale, this.minScale, this.maxScale);
    if (Math.abs(newScale - this.scale) < 1e-4) return;
    const ratio = newScale / this.scale;
    this.tx = cx - (cx - this.tx) * ratio;
    this.ty = cy - (cy - this.ty) * ratio;
    this.scale = newScale;
    this._clampPan();
    this._apply();
  }

  zoomIn()  { this.zoomAt(this.scale * this.zoomStep); }
  zoomOut() { this.zoomAt(this.scale / this.zoomStep); }
  reset()   {
    this.scale = 1; this.tx = 0; this.ty = 0;
    this._apply();
  }

  // ─── Wiring ──────────────────────────────────────────────────
  _attach() {
    for (const p of this.panes) {
      p.container.addEventListener('wheel', (e) => this._onWheel(e, p), { passive: false });
      p.container.addEventListener('pointerdown', (e) => this._onPointerDown(e, p));
      p.container.addEventListener('dblclick', () => this.reset());
      p.canvas.style.transformOrigin = 'center center';
      p.canvas.style.willChange = 'transform';
      p.canvas.style.cursor = 'grab';
    }
    window.addEventListener('keydown', (e) => this._onKey(e));
  }

  _onWheel(e, pane) {
    e.preventDefault();
    const rect = pane.container.getBoundingClientRect();
    // cursor coords relative to container CENTER (matches transform-origin)
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    const factor = Math.exp(-e.deltaY * 0.0015);
    this.zoomAt(this.scale * factor, cx, cy);
  }

  _onPointerDown(e, pane) {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    e.preventDefault();
    pane.canvas.style.cursor = 'grabbing';
    pane.container.setPointerCapture(e.pointerId);

    const startX = e.clientX, startY = e.clientY;
    const startTx = this.tx, startTy = this.ty;

    const move = (ev) => {
      this.tx = startTx + (ev.clientX - startX);
      this.ty = startTy + (ev.clientY - startY);
      this._clampPan();
      this._apply();
    };
    const up = (ev) => {
      pane.canvas.style.cursor = 'grab';
      try { pane.container.releasePointerCapture(ev.pointerId); } catch {}
      pane.container.removeEventListener('pointermove', move);
      pane.container.removeEventListener('pointerup', up);
      pane.container.removeEventListener('pointercancel', up);
    };
    pane.container.addEventListener('pointermove', move);
    pane.container.addEventListener('pointerup', up);
    pane.container.addEventListener('pointercancel', up);
  }

  _onKey(e) {
    // Don't hijack typing in form fields
    const t = e.target;
    if (t && t.matches && t.matches('input, textarea, select')) return;
    if (e.key === '0')                           { this.reset(); }
    else if (e.key === '+' || e.key === '=')     { this.zoomIn(); }
    else if (e.key === '-' || e.key === '_')     { this.zoomOut(); }
    else { return; }
    e.preventDefault();
  }

  // ─── Apply / constrain ───────────────────────────────────────
  _apply() {
    const t = `translate(${this.tx.toFixed(1)}px, ${this.ty.toFixed(1)}px) scale(${this.scale.toFixed(4)})`;
    for (const p of this.panes) p.canvas.style.transform = t;
    this.onChange(this.scale);
  }

  /**
   * Loose pan limits so the user can't fling the image entirely off-screen.
   * Allows panning up to half the *displayed* canvas size beyond the container
   * edge — generous, but always keeps part of the image visible.
   */
  _clampPan() {
    if (this.scale <= 1) {
      this.tx = 0; this.ty = 0;
      return;
    }
    const pane = this.panes[0];
    if (!pane || !pane.canvas) return;
    const rect = pane.container.getBoundingClientRect();
    const cw = pane.canvas.clientWidth * this.scale;
    const ch = pane.canvas.clientHeight * this.scale;
    const xLim = Math.max(0, (cw - rect.width) / 2 + rect.width / 2);
    const yLim = Math.max(0, (ch - rect.height) / 2 + rect.height / 2);
    this.tx = clamp(this.tx, -xLim, xLim);
    this.ty = clamp(this.ty, -yLim, yLim);
  }
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
