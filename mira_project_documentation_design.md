# MIRA Design & Technical Blueprint

## 1. Vision & North Star: "The Clinical Atelier"
MIRA is designed to bridge the gap between high-precision medical computing and intuitive human clinical review. The aesthetic is built on **Clinical Precision**, **Diagnostic Certainty**, and **Human-Centric Safety**.

---

## 2. Visual Foundation (Design Tokens)

### Color Palette
*   **Deep Space (Primary BG):** `#0F1514` — Reduces eye strain for radiologists in dark reading rooms.
*   **Atelier Teal (Primary Action):** `#94D3C1` — A sophisticated, high-visibility clinical green.
*   **Surface High (Cards/Panels):** `#171D1C` — Subtle tonal layering for hierarchy.
*   **Subtle Gray (Body Text):** `#BFC9C4` — High legibility with reduced glare.
*   **Warning Amber (Caution):** For suboptimal pipeline methods.
*   **Success Emerald (Optimal):** For "Optimal Method" indicators.

### Typography
*   **Font Family:** `Manrope` (Variable Weight)
*   **Headlines:** Semi-bold, tight tracking (`-0.02em`), uppercase for labels.
*   **Body:** Medium weight for legibility, generous line-height (`1.6`).

### Shape & Surface
*   **Radius:** `8px` (Small) for a sharp, technical feel.
*   **Borders:** `0.5px` stroke with low opacity for subtle definition.
*   **Elevation:** Flat design with tonal layering rather than heavy shadows.

---

## 3. Core Component Architecture

### A. The Split-Panel Viewer
*   **Left Panel (Source):** Original high-res retinal fundus imagery. Includes metadata overlay (RES, DPI, CH).
*   **Right Panel (Output):** Processed result. Overlay layers (Vessels/Lesions) can be toggled.
*   **Interaction:** Synced zooming and panning between both panels.

### B. Pipeline Configuration (The "Learning" Bar)
*   **Logic:** 4 distinct stages (Channel Extraction, Contrast, Segmentation, Lesion Highlight).
*   **Interaction:** 
    *   **Pills:** Toggle buttons to switch methods.
    *   **Tags:** Status labels (Optimal / Modified / Suboptimal).
    *   **Rationale Box:** A contextual info box that appears when a method is selected, explaining clinical impact.

---

## 4. Functional Logic & User Flow

### Flow 1: The Standard Diagnostic Path
1. **Upload:** User provides raw image.
2. **Auto-Config:** System pre-selects the "Optimal" pipeline defaults based on image entropy.
3. **Run:** Processing bar animates sequentially through the 4 stages.
4. **Review:** Results appear in the right panel with mapped diagnostic insights.

### Flow 2: The Learning/Explore Path
1. **Selection:** User taps a method pill (e.g., swapping "Green Channel" for "Red Channel").
2. **Contextual Feedback:** Pipeline steps panel updates immediately with a "Rationale" explaining why this choice might be suboptimal.
3. **Re-run:** "Run Pipeline" changes to "Re-run Pipeline".
4. **Comparison:** User compares the visual delta between the optimal and modified outputs.

---

## 5. Implementation Notes for Re-creation
*   **Layout:** Use CSS Grid for the split-panel and Flexbox for the pipeline steps.
*   **State Management:** Treat the `pipelineConfig` as a single object state. When any pill is clicked, update the object, which triggers a UI re-render of the explanations and the "Optimal" status tags.
*   **Glassmorphism:** Use `backdrop-blur-xl` and semi-transparent backgrounds for the overlay controls to keep the focus on the medical imagery.