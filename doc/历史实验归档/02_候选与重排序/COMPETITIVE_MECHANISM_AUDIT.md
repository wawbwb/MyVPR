# Fixed-checkpoint CD-BoQ mechanism diagnosis

The three-arm screen finished. Selected epochs are all3. MSLS correct counts:
RU675, temperature669, competitive676, shuffled673. Pitts counts: RU7160,
temperature7148, competitive7157, shuffled7163. This is not stable improvement.

Next step approved by the user: no optimization, strength sweep or seed expansion.
Use each arm's existing GSV-selected best.pt, verified against completed-file
hashes, original checkpoint hash, training code and metadata. Reuse one frozen
backbone extraction per query across all four variants. Keep all original files.

Measure ALL740 MSLS and7608 Pitts query images, with no outcome-based sampling.
Report within-head query attention cosine overlap, normalized entropy and
pre-output-LayerNorm slot singular-value-entropy effective rank per BoQ block.
Also measure final normalized descriptor L2 drift. Save per-query arrays.

Primary comparison groups use competitive-vs-RU correctness for ALL arms so
group membership is held fixed: corrections, regressions, stable correct,
stable error, and all. Secondary reports use each arm's own outcomes. Preserve
query IDs; mean and median are descriptive only, small subgroups especially.

Retrieval outcomes come from the previously verified full-database evaluation;
this audit does NOT recompute retrieval or diagnose database-side changes.
Query-only correlation cannot establish causation. Compare the direction of
overlap/rank changes across groups before interpreting improved diversity as
useful information. No automatic pass threshold and no automatic training.

Remote command: `bash scripts/run_competitive_mechanism.sh` (physical GPU1).
Output: `doc/competitive_mechanism_v1`. Resume preserves completed dataset
arrays; an interrupted current dataset is re-extracted. Small result cache only.
