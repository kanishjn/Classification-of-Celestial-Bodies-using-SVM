#!/usr/bin/env python3
"""
demo.py — Real-time SDSS Astronomical Object Classifier
--------------------------------------------------------
Reuses the existing pipeline directly from:
  • preprocess_and_extract.py  → load_fits, background_subtract,
                                  segment_object, extract_features_from_image
  • train_model_improved.py    → ImprovedSVMTrainer.engineer_advanced_features

Usage:
    python3 demo.py                                      # interactive prompt
    python3 demo.py sdss_data/images/GALAXY_10.fits      # single image
    python3 demo.py sdss_data/images/QSO_1.fits sdss_data/images/STAR_0.fits
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import joblib

# ── reuse existing pipeline code — no duplication ────────────────────────────
from preprocess_and_extract import (
    load_fits,
    background_subtract,
    segment_object,
    extract_features_from_image,   # produces the 35 base features
)
from train_model_improved import ImprovedSVMTrainer

# Stateless trainer — only engineer_advanced_features() is called on it
_trainer = ImprovedSVMTrainer(random_state=42)

# ── colours for each class ────────────────────────────────────────────────────
CLASS_COLOURS = {
    'GALAXY': '#4CAF50',   # green
    'QSO':    '#FF9800',   # orange
    'STAR':   '#2196F3',   # blue
}
CLASS_EMOJIS = {
    'GALAXY': '🌌',
    'QSO':    '✨',
    'STAR':   '⭐',
}

# ── full prediction pipeline ──────────────────────────────────────────────────

def predict_fits(fpath, pipeline, label_encoder):
    """
    FITS file → class prediction + confidence scores.

    Steps (all reusing existing module functions):
      1. load_fits                    — astropy FITS loader
      2. background_subtract          — median sky subtraction
      3. segment_object               — adaptive threshold + largest component
      4. extract_features_from_image  — 35 base features
      5. engineer_advanced_features   — 35 → 61 engineered features
      6. pipeline.predict / predict_proba — SVM inference

    Returns (result_dict, error_string).
    """
    t0 = time.perf_counter()

    img    = load_fits(fpath)
    img_bs = background_subtract(img, boxsize=40)
    mask   = segment_object(img_bs)

    if mask is None:
        return None, "Segmentation failed — object could not be isolated"

    base = extract_features_from_image(img_bs, mask)   # (35,)
    if base is None:
        return None, "Feature extraction failed"

    # engineer_advanced_features expects shape (n_samples, 35)
    X_eng = _trainer.engineer_advanced_features(base.reshape(1, -1), verbose=False)

    pred  = pipeline.predict(X_eng)           # encoded int label
    proba = pipeline.predict_proba(X_eng)[0]  # (n_classes,)

    class_name  = label_encoder.inverse_transform(pred)[0]
    class_probs = {
        label_encoder.classes_[i]: float(proba[i])
        for i in range(len(label_encoder.classes_))
    }
    elapsed = (time.perf_counter() - t0) * 1000   # ms

    # ground truth from filename convention  e.g. GALAXY_10.fits
    fname = os.path.basename(fpath)
    gt    = fname.split('_')[0].upper()
    known = gt in ('GALAXY', 'QSO', 'STAR')

    return {
        'class':      class_name,
        'probs':      class_probs,
        'elapsed_ms': elapsed,
        'img_raw':    img,
        'img_bs':     img_bs,
        'mask':       mask,
        'fname':      fname,
        'gt':         gt if known else None,
    }, None

# ── visualisation ─────────────────────────────────────────────────────────────

def build_confidence_bar(ax, class_probs, predicted):
    classes = sorted(class_probs.keys())
    values  = [class_probs[c] for c in classes]
    colours = [CLASS_COLOURS.get(c, '#888') for c in classes]
    bars = ax.barh(classes, values, color=colours, edgecolor='white',
                   linewidth=1.5, height=0.55)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_title('Model Confidence', fontsize=12, fontweight='bold', pad=8)
    ax.axvline(x=0.33, color='gray', linestyle='--', alpha=0.4, linewidth=1)

    for bar, cls, val in zip(bars, classes, values):
        marker = ' ◀ PREDICTION' if cls == predicted else ''
        ax.text(min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}{marker}',
                va='center', ha='left', fontsize=10,
                fontweight='bold' if cls == predicted else 'normal')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.xaxis.set_tick_params(labelcolor='white')
    ax.yaxis.set_tick_params(labelcolor='white', labelsize=11)

def scale_img(img):
    p1, p99 = np.percentile(img, (1, 99))
    if p99 - p1 > 0:
        return np.clip((img - p1) / (p99 - p1), 0, 1)
    return np.clip(img - p1, 0, 1)

def show_result(result):
    cls    = result['class']
    probs  = result['probs']
    elapsed = result['elapsed_ms']
    gt     = result['gt']
    fname  = result['fname']
    colour = CLASS_COLOURS.get(cls, '#888')
    emoji  = CLASS_EMOJIS.get(cls, '❓')
    correct = (gt == cls) if gt else None

    fig = plt.figure(figsize=(16, 7), facecolor='#0d1117')
    fig.patch.set_facecolor('#0d1117')

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.04, right=0.98,
                           top=0.88, bottom=0.08)

    # ── Panel 0: raw image ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(scale_img(result['img_raw']), cmap='gray', origin='lower')
    ax0.set_title('Raw FITS Image', color='white', fontsize=11, pad=6)
    ax0.axis('off')

    # ── Panel 1: background-subtracted ───────────────────────────────────
    ax1 = fig.add_subplot(gs[:, 1])
    ax1.imshow(scale_img(result['img_bs']), cmap='inferno', origin='lower')
    ax1.set_title('Background Subtracted', color='white', fontsize=11, pad=6)
    ax1.axis('off')

    # ── Panel 2: isolated object (black background) + contour outline ────
    ax2 = fig.add_subplot(gs[:, 2])
    scaled_bs  = scale_img(result['img_bs'])
    mask_bool  = result['mask'] > 0

    # Show only the masked object pixels; everything else → black
    isolated = np.zeros_like(scaled_bs)
    isolated[mask_bool] = scaled_bs[mask_bool]

    # Build an RGB: object in 'inferno' colourmap, background pure black
    import matplotlib.cm as cm
    coloured = cm.inferno(isolated)          # RGBA (H, W, 4)
    coloured[~mask_bool] = [0, 0, 0, 1]     # force background to black

    ax2.imshow(coloured, origin='lower')

    # Draw a bright contour around the detected boundary
    ax2.contour(mask_bool.astype(float), levels=[0.5],
                colors=['#00FFFF'], linewidths=[1.5])

    ax2.set_title('Segmented Object', color='white', fontsize=11, pad=6)
    ax2.axis('off')

    # ── Panel 3 top: confidence bar chart ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor('#0d1117')
    build_confidence_bar(ax3, probs, cls)

    # ── Panel 3 bottom: prediction card ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.set_facecolor('#0d1117')
    ax4.axis('off')

    # Outer glow box
    fancy = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                           boxstyle='round,pad=0.04',
                           linewidth=2.5,
                           edgecolor=colour,
                           facecolor=colour + '22')
    ax4.add_patch(fancy)

    ax4.text(0.5, 0.82, f'{emoji}  PREDICTION', transform=ax4.transAxes,
             ha='center', va='center', fontsize=10, color='#aaa')
    ax4.text(0.5, 0.60, cls, transform=ax4.transAxes,
             ha='center', va='center', fontsize=22,
             fontweight='bold', color=colour)
    ax4.text(0.5, 0.40, f'{probs[cls]:.1%} confidence',
             transform=ax4.transAxes,
             ha='center', va='center', fontsize=12, color='white')

    if gt is not None:
        tick  = '✅ CORRECT' if correct else f'❌ Expected: {gt}'
        col2  = '#4CAF50' if correct else '#f44336'
        ax4.text(0.5, 0.20, tick, transform=ax4.transAxes,
                 ha='center', va='center', fontsize=10,
                 color=col2, fontweight='bold')

    ax4.text(0.5, 0.06, f'⚡ {elapsed:.1f} ms inference',
             transform=ax4.transAxes,
             ha='center', va='center', fontsize=9, color='#666')

    # ── main title ────────────────────────────────────────────────────────
    fig.suptitle(f'SDSS Astronomical Classifier  —  {fname}',
                 fontsize=13, color='white', fontweight='bold', y=0.97)

    plt.show()

# ── CLI entry point ───────────────────────────────────────────────────────────

def run_demo(fpath, pipeline, label_encoder):
    print(f'\n{"═"*60}')
    print(f'  🔭  Classifying: {os.path.basename(fpath)}')
    print(f'{"═"*60}')

    result, err = predict_fits(fpath, pipeline, label_encoder)

    if err:
        print(f'  ❌  Error: {err}')
        return

    cls     = result['class']
    probs   = result['probs']
    elapsed = result['elapsed_ms']
    gt      = result['gt']

    print(f'\n  Prediction  :  {CLASS_EMOJIS.get(cls,"")}  {cls}')
    print(f'  Confidence  :  {probs[cls]:.1%}')
    print(f'  Inference   :  {elapsed:.1f} ms\n')
    print('  Class probabilities:')
    for c in sorted(probs, key=probs.get, reverse=True):
        bar = '█' * int(probs[c] * 30)
        print(f'    {c:<8} {bar:<30} {probs[c]:.1%}')

    if gt:
        correct = gt == cls
        verdict = '✅ CORRECT' if correct else f'❌ Wrong  (true label: {gt})'
        print(f'\n  Ground truth: {gt}  →  {verdict}')

    show_result(result)

def main():
    # ── load model ────────────────────────────────────────────────────────
    model_path = 'improved_svm_model.joblib'
    if not os.path.exists(model_path):
        print(f'❌ Model file not found: {model_path}')
        print('   Run  python3 train_model_improved.py  first.')
        sys.exit(1)

    print('🚀 Loading model…', end=' ', flush=True)
    m = joblib.load(model_path)
    pipeline      = m['pipeline']
    label_encoder = m['label_encoder']
    print('done.')

    # ── resolve file path ─────────────────────────────────────────────────
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        print('\nEnter the path to a FITS file (or drag-and-drop it here):')
        raw = input('  › ').strip().strip("'\"")
        paths = [raw]

    for fpath in paths:
        fpath = fpath.strip().strip("'\"")
        if not os.path.exists(fpath):
            print(f'❌ File not found: {fpath}')
            continue
        run_demo(fpath, pipeline, label_encoder)

        if len(paths) == 1:
            again = input('\nClassify another image? [y/N] › ').strip().lower()
            while again == 'y':
                raw = input('  Path › ').strip().strip("'\"")
                if not os.path.exists(raw):
                    print(f'❌ File not found: {raw}')
                else:
                    run_demo(raw, pipeline, label_encoder)
                again = input('\nClassify another? [y/N] › ').strip().lower()

if __name__ == '__main__':
    main()
