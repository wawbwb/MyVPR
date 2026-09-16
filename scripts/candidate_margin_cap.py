"""List loss + capped safety-margin preservation; reuse the immutable runner."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import candidate_margin_preserve as runner

CAP = 1.0
BASE_CODES = runner.codes
POLICY = dict(runner.POLICY, preservation_cap=CAP,
    preservation='mean over ALL batch queries of frozen-correct * relu(min(detached frozen margin, 1.0) - current margin)',
    experiment='original list loss versus list loss plus capped preservation, weight 1; preserve arm means CAPPED in this run',
    scope='Post-hoc exploratory capped-margin objective ablation. No heldout threshold tuning or Pitts access.')


def codes():
    return {**BASE_CODES(), 'scripts/candidate_margin_cap.py': hashlib.sha256(
        Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}


def objective(scores, base, labels, weight):
    import torch.nn.functional as F
    valid = labels.any(-1) & (~labels).any(-1)
    if not bool(valid.all()): raise ValueError('Training batch must have positive and negative candidates')
    pos = scores.masked_fill(~labels, float('-inf'))
    neg = scores.masked_fill(labels, float('-inf'))
    retrieval = F.softplus(neg.logsumexp(-1)-pos.logsumexp(-1)).mean()
    frozen = base.detach()
    correct = labels.gather(1, frozen.argmax(-1, keepdim=True)).squeeze(1)
    frozen_margin = (frozen.masked_fill(~labels, float('-inf')).max(-1).values
                     - frozen.masked_fill(labels, float('-inf')).max(-1).values)
    target = frozen_margin.clamp(max=CAP)
    margin = pos.max(-1).values-neg.max(-1).values
    penalty = (F.relu(target-margin)*correct.to(scores.dtype)).mean()
    return retrieval+weight*penalty, retrieval, penalty, correct.sum()


def main(argv=None):
    # Reuse exactly the same optimizer, ordering, augmentation, checkpointing,
    # diagnostics and resume machinery. Runtime substitution is scoped/restored;
    # no old source or cache contract is edited or bypassed.
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(x == '--output' or x.startswith('--output=') for x in args):
        args += ['--output', 'logs/candidate_margin_cap_v1']
    old = (runner.objective, runner.POLICY, runner.codes, sys.argv)
    try:
        runner.objective, runner.POLICY, runner.codes = objective, POLICY, codes
        sys.argv = [str(Path(__file__))]+args
        runner.main()
    finally:
        runner.objective, runner.POLICY, runner.codes, sys.argv = old


if __name__ == '__main__': main()
