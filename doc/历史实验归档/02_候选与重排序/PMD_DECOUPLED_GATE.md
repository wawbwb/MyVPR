# Decoupled correspondence / matchability diagnostic

No VPR training or efficacy claim. Same256 GSV train images /64 dev images,
known crop/flip/occlusion labels as correspondence_v1,3 epochs,seed42,LR1e-4.
Historical dev, not independent confirmation. All runs are new; old runs retained.

Three arms: forced balanced matching; joint partial transport+dustbin; forced
matching plus detached rejection MLP. Identical norm/key initialization. Rejection
MLP sees projected source feature, transported target feature, absolute difference,
maximum assignment probability and entropy. Its inputs are detached; gradients
and clipping are separated from the matcher. Added MLP has24833 parameters:
this is NOT an exactly parameter-matched architecture comparison. Decoder frozen.

Use200 Sinkhorn iterations in all transport arms, explicit dustbin probability
and normalized real+dustbin mass. Forced real rows normalized; its rejection is
identically zero. Loss is matched position NLL plus equally weighted positive/
negative binary rejection NLL where applicable. Partial position NLL includes
matched mass; forced/decoupled location NLL is conditional. This difference is
part of the hypothesis, not identical objectives. Compare against re-run controls,
not old20-iteration results. Fixed rejection probability threshold0.5; no dev
threshold fitting, epoch selection or extra training based on results.

Primary checks: decoupled must retain forced location accuracy and improve
rejection trade-off relative to joint partial. Report location accuracy, final
matched accuracy, false rejection and unmatched rejection recall separately.
No automatic pass declaration, no assumption of cross-view/general VPR benefit.
Same-source synthetic views may allow shortcuts; later real-view tests remain
required. Do not interpret correlated patch counts as independent sample size.

```bash
python -m pytest -q tests/test_pmd_decoupled.py
bash scripts/run_pmd_decoupled.sh
```

Supports deterministic cursor resume, checkpoints every16 examples. Stores
small checkpoints/metrics, no dense feature cache. Tests run on training machine.
