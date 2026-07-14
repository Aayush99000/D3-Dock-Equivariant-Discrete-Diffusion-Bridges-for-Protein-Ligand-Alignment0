#!/usr/bin/env python3
"""
D3-Dock Demo Viewer.

Generates a self-contained HTML file for a single protein-ligand system showing:
  Tab A — Animated diffusion: ligand morphs from random noise → predicted pose
           while the protein pocket stays fixed in the background.
  Tab B — Static overlay: protein cartoon + predicted ligand (orange) + true
           ligand (green) for direct comparison.

Uses 3Dmol.js (loaded from CDN) — no local dependencies needed, works in any browser.

Usage:
    python scripts/demo_viewer.py \
        --plinder-id   2ci5__1__1.A__1.C \
        --protein-pdb  /scratch/.../preprocessed/2ci5__1__1.A__1.C/2ci5__1__1.A__1.C.clean.pdb \
        --true-sdf     /scratch/.../preprocessed/2ci5__1__1.A__1.C/2ci5__1__1.A__1.C.rdkit.sdf \
        --pred-sdf     /scratch/.../eval_val_v4_T200/pred_sdf/2ci5__1__1.A__1.C.pred.sdf \
        --trajectory   /scratch/.../eval_demo/trajectory/2ci5__1__1.A__1.C_trajectory.npz \
        --rmsd         1.949 \
        --output       ./outputs/demo/2ci5__1__1.A__1.C_demo.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate D3-Dock interactive HTML demo.")
    p.add_argument("--plinder-id",   required=True)
    p.add_argument("--protein-pdb",  required=True)
    p.add_argument("--true-sdf",     required=True)
    p.add_argument("--pred-sdf",     required=True)
    p.add_argument("--trajectory",   required=True, help="NPZ file from --save-trajectory.")
    p.add_argument("--rmsd",         type=float, default=None, help="RMSD vs ground truth (Å).")
    p.add_argument("--output",       required=True, help="Output HTML file path.")
    return p.parse_args()


def _read_text(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


def _frames_to_json(npz_path: str) -> str:
    """Load trajectory NPZ and return JSON list of world-space frames.

    Each frame is a list of {x, y, z} dicts — one per ligand heavy atom.
    Frames run from t=T (pure noise) → t=0 (predicted pose).
    """
    d = np.load(npz_path)
    frames = d["frames"].astype(float)   # (F, N, 3) COM-normalised
    com = d["com"].astype(float)          # (3,)
    # Add COM to recover world-space coordinates
    frames_world = frames + com[None, None, :]  # (F, N, 3)

    result = []
    for frame in frames_world:
        result.append([{"x": float(a[0]), "y": float(a[1]), "z": float(a[2])} for a in frame])
    return json.dumps(result)


def generate_html(
    plinder_id: str,
    protein_pdb_str: str,
    true_sdf_str: str,
    pred_sdf_str: str,
    frames_json: str,
    rmsd: float | None,
) -> str:
    rmsd_str = f"{rmsd:.3f} Å" if rmsd is not None else "N/A"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>D3-Dock Demo — {plinder_id}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f1117; color: #e0e0e0; }}
  header {{ padding: 18px 28px; background: #1a1d27; border-bottom: 1px solid #2a2d3a; }}
  header h1 {{ font-size: 1.2rem; font-weight: 600; color: #fff; }}
  header p  {{ font-size: 0.82rem; color: #888; margin-top: 4px; }}
  .badge    {{ display: inline-block; background: #1e7e34; color: #fff;
               font-size: 0.75rem; font-weight: 700; border-radius: 4px;
               padding: 2px 8px; margin-left: 10px; vertical-align: middle; }}
  .tabs     {{ display: flex; gap: 4px; padding: 14px 28px 0; background: #1a1d27; }}
  .tab-btn  {{ padding: 8px 22px; border: none; border-radius: 6px 6px 0 0;
               background: #252836; color: #888; cursor: pointer; font-size: 0.88rem;
               font-weight: 500; transition: background 0.2s, color 0.2s; }}
  .tab-btn.active {{ background: #2d3250; color: #fff; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: flex; flex-direction: column; height: calc(100vh - 130px); }}
  .viewer-wrap {{ flex: 1; position: relative; }}
  .viewer-box  {{ width: 100%; height: 100%; }}
  .controls    {{ padding: 12px 20px; background: #1a1d27; border-top: 1px solid #2a2d3a;
                  display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
  .btn         {{ padding: 6px 18px; border: none; border-radius: 5px; cursor: pointer;
                  font-size: 0.83rem; font-weight: 600; transition: opacity 0.15s; }}
  .btn:hover   {{ opacity: 0.85; }}
  .btn-play    {{ background: #4f9cf9; color: #fff; }}
  .btn-reset   {{ background: #3a3d4e; color: #ccc; }}
  .progress    {{ flex: 1; min-width: 120px; accent-color: #4f9cf9; }}
  .step-label  {{ font-size: 0.8rem; color: #666; min-width: 90px; }}
  .legend      {{ display: flex; gap: 16px; align-items: center; }}
  .legend-dot  {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block;
                  margin-right: 5px; }}
</style>
</head>
<body>

<header>
  <h1>D3-Dock &mdash; Reverse Diffusion Docking Demo
    {f'<span class="badge">RMSD {rmsd_str}</span>' if rmsd is not None else ""}
  </h1>
  <p>System: <code>{plinder_id}</code> &nbsp;|&nbsp;
     Model: v4 (128-dim, radius graph, 12 798 training systems, ep 200)
  </p>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('anim', this)">
    ▶ Diffusion Animation
  </button>
  <button class="tab-btn" onclick="switchTab('static', this)">
    ⬡ Static Overlay
  </button>
</div>

<!-- ── TAB A: Animation ──────────────────────────────────────────────────── -->
<div id="tab-anim" class="tab-panel active">
  <div class="viewer-wrap">
    <div id="anim-viewer" class="viewer-box"></div>
  </div>
  <div class="controls">
    <button class="btn btn-play"  id="play-btn"  onclick="togglePlay()">▶ Play</button>
    <button class="btn btn-reset" onclick="resetAnim()">↺ Reset</button>
    <input  type="range" class="progress" id="frame-slider" min="0" value="0"
            oninput="goToFrame(parseInt(this.value))">
    <span   class="step-label" id="step-label">Step T (noise)</span>
    <div class="legend">
      <span><span class="legend-dot" style="background:#ff6b35"></span>Predicted pose</span>
    </div>
  </div>
</div>

<!-- ── TAB B: Static overlay ─────────────────────────────────────────────── -->
<div id="tab-static" class="tab-panel">
  <div class="viewer-wrap">
    <div id="static-viewer" class="viewer-box"></div>
  </div>
  <div class="controls">
    <div class="legend">
      <span><span class="legend-dot" style="background:#27ae60"></span>True pose (ground truth)</span>
      <span><span class="legend-dot" style="background:#ff6b35"></span>Predicted pose</span>
      <span><span class="legend-dot" style="background:#4f9cf9;border-radius:2px"></span>Protein surface</span>
    </div>
    <span style="font-size:0.82rem;color:#666;margin-left:auto;">
      RMSD: <strong style="color:#fff">{rmsd_str}</strong>
    </span>
  </div>
</div>

<script>
// ── Embedded data ────────────────────────────────────────────────────────────
const PROTEIN_PDB = {json.dumps(protein_pdb_str)};
const TRUE_SDF    = {json.dumps(true_sdf_str)};
const PRED_SDF    = {json.dumps(pred_sdf_str)};
const FRAMES      = {frames_json};   // array of frames, each = [{{x,y,z}}, ...]

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(id, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'static' && !staticReady) initStatic();
}}

// ── Animation viewer ─────────────────────────────────────────────────────────
let animViewer, animInterval, currentFrame = 0, playing = false;
const N_FRAMES = FRAMES.length;

function initAnim() {{
  animViewer = $3Dmol.createViewer(document.getElementById('anim-viewer'), {{
    backgroundColor: '#0f1117'
  }});

  // Protein — surface representation, semi-transparent blue
  animViewer.addModel(PROTEIN_PDB, 'pdb');
  animViewer.setStyle(
    {{ model: 0 }},
    {{ cartoon: {{ color: '#4f9cf9', opacity: 0.25 }},
       surface: {{ opacity: 0.10, color: '#4f9cf9' }} }}
  );

  // Ligand — starts as noise frame
  animViewer.addModel('', 'sdf');   // placeholder model index 1
  animViewer.setStyle({{ model: 1 }}, {{ sphere: {{ radius: 0.35, color: '#ff6b35' }} }});

  document.getElementById('frame-slider').max = N_FRAMES - 1;
  goToFrame(0);
  animViewer.zoomTo();
  animViewer.render();
}}

function goToFrame(f) {{
  currentFrame = f;
  document.getElementById('frame-slider').value = f;
  const isNoise = (f === 0);
  const isClean = (f === N_FRAMES - 1);
  const stepNum  = isNoise ? 'T (noise)' : isClean ? '0 (pose)' : `${{Math.round((1 - f/(N_FRAMES-1)) * {frames_json.count('[') - 1 if '[' in frames_json else 200})}}`;
  document.getElementById('step-label').textContent = 'Step ' + stepNum;

  // Rebuild ligand model from current frame coords
  const atoms = FRAMES[f];
  const lines = [''];
  atoms.forEach((a, i) => {{
    const xi = a.x.toFixed(4).padStart(10);
    const yi = a.y.toFixed(4).padStart(10);
    const zi = a.z.toFixed(4).padStart(10);
    lines.push(`C   ${{xi}} ${{yi}} ${{zi}}  1.00  0.00           C`);
  }});
  // Remove and re-add ligand model
  animViewer.removeModel(animViewer.getModel(1));
  const sdfBlock = buildSdfFromCoords(atoms);
  animViewer.addModel(sdfBlock, 'sdf');
  animViewer.setStyle({{ model: 1 }}, {{
    sphere: {{ radius: 0.35, color: '#ff6b35', opacity: isNoise ? 0.5 : 1.0 }}
  }});
  animViewer.render();
}}

function buildSdfFromCoords(atoms) {{
  // Minimal V2000 SDF with correct atom count for rendering
  const n = atoms.length;
  let sdf = '\\n  D3Dock\\n\\n';
  sdf += String(n).padStart(3) + '  0  0  0  0  0  0  0  0  0999 V2000\\n';
  atoms.forEach(a => {{
    sdf += a.x.toFixed(4).padStart(10) + a.y.toFixed(4).padStart(10) +
           a.z.toFixed(4).padStart(10) + ' C   0  0  0  0  0  0  0  0  0  0  0  0\\n';
  }});
  sdf += 'M  END\\n$$$$\\n';
  return sdf;
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {{
    animInterval = setInterval(() => {{
      const next = (currentFrame + 1) % N_FRAMES;
      goToFrame(next);
      if (next === N_FRAMES - 1) {{ playing = false; clearInterval(animInterval);
        document.getElementById('play-btn').textContent = '▶ Play'; }}
    }}, 120);
  }} else {{
    clearInterval(animInterval);
  }}
}}

function resetAnim() {{
  playing = false;
  clearInterval(animInterval);
  document.getElementById('play-btn').textContent = '▶ Play';
  goToFrame(0);
}}

// ── Static overlay viewer ────────────────────────────────────────────────────
let staticReady = false;
function initStatic() {{
  staticReady = true;
  const sv = $3Dmol.createViewer(document.getElementById('static-viewer'), {{
    backgroundColor: '#0f1117'
  }});

  // Protein
  sv.addModel(PROTEIN_PDB, 'pdb');
  sv.setStyle({{ model: 0 }}, {{
    cartoon: {{ color: '#4f9cf9', opacity: 0.5 }},
    surface: {{ opacity: 0.08, color: '#4f9cf9' }}
  }});

  // True ligand — green sticks+spheres
  sv.addModel(TRUE_SDF, 'sdf');
  sv.setStyle({{ model: 1 }}, {{
    stick:   {{ radius: 0.18, color: '#27ae60' }},
    sphere:  {{ radius: 0.32, color: '#27ae60' }}
  }});

  // Predicted ligand — orange sticks+spheres
  sv.addModel(PRED_SDF, 'sdf');
  sv.setStyle({{ model: 2 }}, {{
    stick:   {{ radius: 0.18, color: '#ff6b35' }},
    sphere:  {{ radius: 0.32, color: '#ff6b35' }}
  }});

  sv.zoomTo({{ model: 1 }});
  sv.render();
}}

// ── Init ─────────────────────────────────────────────────────────────────────
$(document).ready(() => {{ initAnim(); }});
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()

    print(f"Loading data for {args.plinder_id}...")
    protein_pdb_str = _read_text(args.protein_pdb)
    true_sdf_str    = _read_text(args.true_sdf)
    pred_sdf_str    = _read_text(args.pred_sdf)
    frames_json     = _frames_to_json(args.trajectory)

    n_frames = len(json.loads(frames_json))
    print(f"  Trajectory frames: {n_frames}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = generate_html(
        plinder_id=args.plinder_id,
        protein_pdb_str=protein_pdb_str,
        true_sdf_str=true_sdf_str,
        pred_sdf_str=pred_sdf_str,
        frames_json=frames_json,
        rmsd=args.rmsd,
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"Demo saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print("Open in any browser — no server needed.")


if __name__ == "__main__":
    main()
