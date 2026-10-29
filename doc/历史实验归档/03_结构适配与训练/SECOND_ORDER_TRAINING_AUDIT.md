# QSO-BoQ training-side diagnostic

The completed three-arm screen did not demonstrate a benefit. Fixed epoch 3:
RU / mean_outer / global_cov / query_cov MSLS correct = 675 / 675 / 675 / 675;
Pitts30k correct = 7160 / 7157 / 7159 / 7158. All GSV-selected epochs were 0.

This post-hoc diagnostic is not a new experiment or a change to model selection.
It loads the verified last checkpoints (epoch 3) and the exact 4096 training
places recorded in the immutable contracts. The 1024 development places are
excluded. Four views per place are sampled with the original epoch-0 per-place
seed, but evaluated with clean resize/normalization, not photometric augmentation.
The frozen backbone is shared across RU and the three heads to minimize cost.

All 16384 views query all other views; self is excluded, same-place views are
positives. Define margin = highest positive similarity - highest other-place
similarity. RU fixes groups: top-1 errors, correct with margin <= 0.02, and remaining
correct. The threshold is a diagnostic convention, not tuned for retrieval gains.
Report query counts, unique place counts, corrections/regressions, margin changes,
and changes on the fixed RU best-positive/hardest-negative pair. This distinguishes
improvement on original hard pairs from changes in which negative is hardest.

Also check whether the RU hardest negative was in the same 16-place batch under
the first training epoch permutation, and the clean within-batch margin. This is
only a **clean epoch-0 batch exposure proxy**: it does not reproduce photometric
augmentations, later sampled views, or the multi-similarity miner. It cannot prove
which gradient each historical batch produced. Training loss zero rates remain
the original recorded measurements, not estimates from this diagnostic.

Interpretation: lack of training-side improvement weakens the current learning
setup; training-side improvement with no development gain suggests a transfer or
generalization issue. Neither result alone proves its cause or rules out all
second-order architectures. Places, not four dependent views, are the natural
units for any subsequent uncertainty analysis. No MSLS/Pitts tuning or retraining.

Training machine only:
```bash
python -m pytest -q tests/test_second_order_training_audit.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/audit_second_order_training.py --checkpoint "$RU_CKPT" --resume
```

Outputs: `doc/second_order_training_audit_v1/{contract,places,ru,mean_outer,global_cov,query_cov,summary,progress,completed}.json`.
No descriptor cache is persisted. Approximately 3.2 GiB of CPU descriptor storage
plus concatenation buffers is needed; GPU scoring is chunked, not a full square
similarity matrix. On interruption, extraction restarts; completed outputs are
hash-verified, and matching partial runs require `--resume`.
