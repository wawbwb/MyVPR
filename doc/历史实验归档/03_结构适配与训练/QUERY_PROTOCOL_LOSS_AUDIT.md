# Paired training objective replay — 2026-10-06

All four protocol arms completed and passed completion hashes/frozen checks.
Fixed epoch3 MSLS/Pitts correct: RU675/7160; mixed1e-4 672/7149;
broad1e-4 675/7158; mixed1e-5 675/7160; broad1e-5 675/7160.
No configuration improved R@1 over RU. Low LR preserved correctness, not a gain.

Next diagnostic (no training): replay all original960 augmented batches per arm
against RU and that arm's fixed epoch3 model. Same MatchedPlaces epoch seeds,
identical images shared by both forward paths, frozen backbone features reused.
Report each epoch and base-broad versus hard/extra-broad slots separately.

Report both each model's own mined loss and trained loss using RU's fixed mined
pairs, mined positive/negative counts and descriptor drift. Loss differences
under changing miners alone are not sufficient evidence of improved geometry.
Cross-epoch RU losses diagnose varying input difficulty; within-batch RU versus
epoch3 differences diagnose final training fit. These are NOT reconstructed
historical online losses, nor independent generalization evidence. No epoch or
hyperparameter selection follows automatically. Preserve source runs unchanged.

Script: `scripts/audit_query_protocol_loss.py --checkpoint ... --resume`.
Output: `logs/query_protocol_loss_audit_v1`. Atomic rows saved every16 batches;
interruption replays at most15 batches. No optimizer or image feature cache.
