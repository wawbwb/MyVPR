# AG-SLRD run archive

`teacher_epoch_summary.csv` is the compact, versionable extraction of the two
formal 10-epoch teacher runs.  It was extracted verbatim from
`teacher_aligned_10ep.txt` and `teacher_shuffled_10ep.txt` on 2026-09-02.
The authoritative final metrics, checkpoint/cache hashes, runtime overflow
events and complete provenance remain embedded in
`../ag_slrd_phase0_audit/summary.json`.

The two raw formal stdout files are valid but mostly consist of tqdm refresh
records and are retained locally as bulky source artifacts.  The file
`teacher_aligned_failed_amp_step6371.txt` is the pre-fix run that stopped in
epoch 5 because the old AMP state machine raised before GradScaler backoff.  It
is an engineering incident record, not a comparable experiment and is not used
for any metric in the Phase-0 verdict.

`phase0_audit.txt` duplicates the compact verdict printed in
`../ag_slrd_phase0_audit/verdict.txt`; the JSON/CSV audit directory is the
canonical result archive.
