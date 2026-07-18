# Experiment Guide — Phases C–F

> **Status (2026-07-18): this C–F plan is archived.** Do not run the D1/D2
> commands below. The active route and commands are documented in
> `doc/VPR_SEMANTIC_RELIABILITY.md`.

Run everything from the project root with the `openvpr` conda environment:

```bash
cd workspace/OpenVPRLab/
conda activate VPR
```

General launch pattern (same as the A/B phases you already ran):

```bash
python run.py --train --config config/<file>.yaml
```

Optional overrides: `--batch_size 80`, `--lr 0.0001`, `--seed 1`,
`--devices 0`. Results
(Recall@K) print per epoch; checkpoints land under
`logs/<backbone>/MixVPR/version_*/` and hyperparameters in that version's
`hparams.yaml`. The metric monitored for the best checkpoint is `msls-val/R1`.

> Before starting, smoke-test the new code path once (no dataset required):
> ```bash
> pytest -q tests/test_phase_c_attention.py
> python scripts/smoke_phase_c.py
> ```
> Then run one real data batch:
> ```bash
> python run.py --dev --config config/mixvpr_distill_attn_kl.yaml
> ```
> `--dev` runs a single train + val iteration (no checkpointing). If it prints
> the data/model summaries without error, the spatial-attention + CLIP-attention
> wiring is good.

Always compare against your **verified baseline** (Exp #1, MSLS R@1 = 87.70).

---

## Phase C — Local Semantic Attention Distillation

**C0 — architecture-only control (student gate, no CLIP teacher)**
```bash
python run.py --train --config config/mixvpr_distill_attn_arch_only.yaml
```
This separates gains from the extra 1x1 spatial-gating capacity from gains due
to CLIP supervision. The C configs deliberately use `lambda_global=0`, so the
first C conclusion is about local semantic information only.

**C2 — student attention head + CLIP attention KL**
```bash
python run.py --train --config config/mixvpr_distill_attn_kl.yaml
```

**C3 — sweep `lambda_kl` ∈ {0.01, 0.05, 0.1, 0.2}**
```bash
python run.py --train --config config/mixvpr_distill_attn_kl_l0.01.yaml
python run.py --train --config config/mixvpr_distill_attn_kl_l0.05.yaml
python run.py --train --config config/mixvpr_distill_attn_kl_l0.1.yaml
python run.py --train --config config/mixvpr_distill_attn_kl_l0.2.yaml
```
Pick the `lambda_kl` that maximises MSLS R@1 without hurting Pitts30k. Use it in
later phases (edit `spatial_attn.lambda_kl`). Only after this pure-local sweep
should you run a separate global+local combination with the best B-stage value.

---

## Phase D — VPR-Oriented Spatial Weight Learning

**D1 — + cross-view consistency**
```bash
python run.py --train --config config/mixvpr_distill_consistency.yaml
```

**D2 — + cross-place divergence**
```bash
python run.py --train --config config/mixvpr_distill_divergence.yaml
```

**D3 — multi-head attention (4 heads)**
```bash
python run.py --train --config config/mixvpr_distill_multihead.yaml
```

Carry forward whichever D-terms help (consistency on/off, divergence on/off,
single vs. 4 heads) into the E-phase config.

---

## Phase E — Full Integration & Tuning

Edit `config/mixvpr_distill_full.yaml` so the lambdas/heads match your best
B/C/D winners, then:

**E1 — best combination, full 40 epochs**
```bash
python run.py --train --config config/mixvpr_distill_full.yaml
```

**E2 — distillation annealing (decay all distill weights, epochs 20→30)**
```bash
python run.py --train --config config/mixvpr_distill_full_anneal.yaml
```

**E3 — ResNet-101 backbone**
```bash
python run.py --train --config config/mixvpr_distill_full_resnet101.yaml
```
(Keep `mixvpr_distill_full_anneal.yaml`/`full_resnet101.yaml` lambdas in sync
with `full.yaml` if you re-tune E1.)

---

## Phase F — Benchmark Validation

These benchmarks are where the semantic prior should actually pay off
(day/night and seasonal change). They are **not bundled** with OpenVPRLab — set
them up first.

**1) Get the datasets and point the config at them.** Edit
`config/data/config.yaml` → `datasets.val`:
```yaml
    tokyo247: /abs/path/to/tokyo247
    nordland: /abs/path/to/nordland
```
Provide the metadata each loader expects (see headers in
`src/dataloaders/valid/tokyo247.py` and `nordland.py`):
- **Tokyo 24/7:** `tokyo247_dbImages.npy`, `tokyo247_qImages.npy`,
  `tokyo247_gt_25m.npy` (references-first, GT = list of db indices within 25 m).
- **Nordland:** either `nordland_dbImages.npy` / `nordland_qImages.npy` /
  `nordland_gt.npy`, **or** two season subfolders (e.g. `summer/`, `winter/`)
  with frame-aligned filenames (the loader builds identity ground truth).

**2) Verify they load** (single dev iteration over all 6 val sets):
```bash
python run.py --dev --config config/mixvpr_distill_full_benchmark.yaml
```

**3) Final comparison.** Run your best E-series checkpoint's config on the full
benchmark suite (`msls-val`, `pitts30k-val`, `msls-val-night`,
`msls-val-season`, `tokyo247`, `nordland`):
```bash
python run.py --train --config config/mixvpr_distill_full_benchmark.yaml
```
For a pure evaluation of an existing checkpoint, also re-run the **A1 baseline**
config with the same extended `val_set_names` so every row in the final table is
directly comparable.

---

## What success looks like

| Benchmark      | Baseline (A1) | Target                 |
| -------------- | ------------- | ---------------------- |
| MSLS-val R@1   | 87.7          | ≥ 87.7 (no regression) |
| Pitts30k R@1   | 93.4          | ≥ 93.4 (no regression) |
| Tokyo 24/7 R@1 | ~70–75        | ≥ 78 (+3–5%)           |
| Nordland R@1   | ~30–40        | ≥ 40–50 (+10%)         |

The semantic distillation should help most where appearance change is extreme
(Tokyo 24/7, Nordland) while staying neutral on the easy benchmarks. If gains
appear **only** on MSLS/Pitts30k, the approach is overfitting to easy data; if
they appear on Tokyo 24/7 / Nordland, the prior is genuinely improving condition
invariance.

---

## Tips
- **Reproducibility:** run each decisive experiment with ≥2 seeds (`--seed`),
  and read the per-epoch curve, not just the best checkpoint — your earlier runs
  peaked around epoch 10–11 then declined.
- **Isolate effects:** the C/D configs use `mode: "global_only"` so the SG-SA
  region branch is off and you measure the attention head's contribution alone.
- **OOM:** drop `--batch_size` to 80/60; keep `img_per_place: 4`.
- **Wrong GPU:** change `devices=[1]` in `run.py`.
