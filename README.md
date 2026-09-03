# CAC-KD: Confidence-Aware Cross-Domain Keypoint Detection for Surgical Instrument Tracking

Official implementation of **CAC-KD**, a confidence-aware framework for surgical
instrument keypoint tracking that uses pseudo-labelled auxiliary data from a
different surgical domain. CAC-KD has three components:

- **CDT** (Confidence-Driven Training) - estimates per-keypoint reliability from
  baseline/pseudo-label agreement and weights the training loss accordingly.
- **CGFA** (Confidence-Gated Feature Adapter) - a 320-parameter channel gate that
  adapts auxiliary-domain features, skipped for target samples.
- **UMGR** (Uncertainty-Modulated Geometry Refinement) - a confidence-modulated
  graph over the instrument skeleton that refines keypoint coordinates (~664 parameters).

Together CGFA and UMGR add only **984 parameters**.

On the cataract benchmark (30-fold LOSO, 1,778 automatically annotated frames),
CAC-KD reduces mean error from 11.54 px to 7.58 px (-34.3%), improving over the
baseline in 25 of 30 folds. On RMIT (cross-sequence), standalone CDT reduces mean
error from 10.72 px to 8.98 px.

## Repository layout

```
src/
  dr_11.py                        # Final training script (all paper results).
                                  # Contains CDT, CGFA and UMGR implementations.
  dr_1.py ... dr_10.py            # Earlier exploration iterations (kept for
                                  # transparency; not needed to reproduce the paper).
  extract_cataract_keypoints.py   # PCA-based pseudo-keypoint extraction from
                                  # Cataract-1K segmentation masks.
  build_rmit_manifest.py,
  generate_manifests.py           # Dataset manifest construction.
  hrnet.py, sota_hrnet_simcc.py,
  resnet.py, mobilenet.py         # Baseline and comparison architectures.
scripts/
  eval_cackd_switch_off.py        # Module switch-off experiment (Table: switch-off).
  eval_instrument_type.py         # Per-instrument-type breakdown.
  analyze_loso_errors.py          # Per-keypoint / anomalous-fold error analysis.
  eval_loso_checkpoint.py, ...    # Other evaluation utilities.
data/
  manifests/                      # RMIT fold manifests.
  cataract_keypoints/
    manifest.json                 # Derived pseudo-keypoint annotations (released).
    visualizations/               # Overlay samples of the extracted keypoints.
```

## Setup

```bash
pip install -r requirements.txt
```

Developed with Python 3.10 and PyTorch (CUDA). Several scripts contain
hard-coded absolute paths (e.g. `/root/autodl-tmp/...` for pretrained weights and
manifests). Adjust these to your local paths before running; they are command-line
arguments or constants near the top of each script.

## Data

Neither dataset is redistributed here.

- **RMIT** (Du et al., 2018): obtain the dataset from the authors of
  "Geometric features-based tracking of instruments in retinal microsurgery".
- **Cataract-1K** (Ghamsarian et al., 2024): obtain the segmentation subset from
  the official release, then rebuild the derived pseudo-keypoints:

```bash
python src/extract_cataract_keypoints.py   # see argparse in the script
python src/generate_manifests.py
```

The derived keypoint annotations themselves are provided in
`data/cataract_keypoints/manifest.json`; `visualizations/` shows overlay samples.
Video frames are **not** redistributed (they belong to the Cataract-1K release).

## Reproducing the paper results

All experiments use `src/dr_11.py`. Example (RMIT LOSO, fold seq1):

```bash
python src/dr_11.py --out_dir runs/loso_seq1 --protocol loso --seq seq1     --manifest data/manifests/rmit_manifest.json
```

Protocol flags: `--protocol loso|within_seq|half` (half-split vs. cross-sequence),
`--seq seq1|seq2|seq3`. `--no_pretrained` disables ImageNet initialisation.
The default hyperparameters correspond to the paper configuration
(sigma 2.0, heatmap+offset decoder, confidence scale alpha = 3).

Evaluation utilities in `scripts/` produce the switch-off, per-instrument and
per-keypoint breakdowns reported in the paper. `PAPER_RESULTS_SUMMARY.md`-style
aggregation was done with `scripts/analyze_loso_errors.py`.

Pretrained surgical backbones (SurgeNetXL, PeskaVLP) follow the original
authors' public releases; see the paper for references.

## Notes

- `dr_1.py` ... `dr_10.py` document the exploration path; `dr_11.py` is the
  paper's final model. They share utility code and are kept for transparency.
- The offline confidence is computed once from the baseline checkpoint before
  CAC-KD training and is fixed during training (see the CDT section of the paper).

## License

MIT - see [LICENSE](LICENSE).

If this code is useful, please cite the paper.
