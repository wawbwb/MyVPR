"""Predeclared ResNet transfer protocol; benefit is an outcome, not assumed."""
import ast
import math
from pathlib import Path

ARMS = ('conv_last', 'conv_cross', 'conditional_last', 'conditional_cross')
POLICY = dict(revision=1, seed=42, arms=list(ARMS), steps=2048, smoke_steps=8,
    train_size=320, eval_sizes=[320, 384], primary_eval_size=384,
    descriptor_dim=16384, native_stages=['layer1', 'layer2', 'layer3'],
    native_channels=[256, 512, 1024], dim=128, rank=32, rho=.025,
    lr=1e-4, weight_decay=0., clip=1., microbatch=2, precision='fp32',
    alignment='Area-pool to layer3 grid, repeat channels to1024 (no learnable parameters); final map unchanged',
    initialization='All arms fresh author ResNet50-BoQ; conv pair identical initialization; conditional pair identical initialization; NO DINO/local/RU warm weights',
    frozen='Entire author encoder+BoQ, including BatchNorm buffers; auxiliary heads frozen/excluded',
    data='Verified historical GSV8192 train-place pool and2048 schedule; epoch99 fixed four views; historical RU used only for negative indices, not features or supervision',
    selection='Fixed last2048, single seed42, no best epoch/size/arm/rho selection; smoke updates discarded',
    scope='Exploratory cross-backbone transfer on exposed MSLS-val and Pitts30k-val; not stable gain, full training, independent test or original-paper test replication')


def paired(reference, candidate):
    if (not reference or len(reference) != len(candidate)
            or any(type(x) is not bool for x in [*reference, *candidate])):
        raise ValueError('Unmatched Boolean query outcome vectors')
    fixes = [i for i, (a, b) in enumerate(zip(reference, candidate)) if not a and b]
    damage = [i for i, (a, b) in enumerate(zip(reference, candidate)) if a and not b]
    n = len(fixes) + len(damage)
    # Stable binomial CDF, even for large mismatch populations.
    k = min(len(fixes), len(damage))
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             - n * math.log(2) for i in range(k + 1)]
    p = min(1., 2 * sum(math.exp(t) for t in terms)) if n else 1.
    return dict(corrections=fixes, regressions=damage, net=len(fixes)-len(damage),
                delta_r1_pp=100*(len(fixes)-len(damage))/len(reference),
                nominal_exact_two_sided_p=p)


def assessment(parts):
    primary = [parts[f'{name}_384'] for name in ('msls-val', 'pitts30k-val')]
    checks = {}
    for arm in ARMS:
        checks[arm + '_beats_official_both'] = all(p[arm]['vs_official']['net'] > 0 for p in primary)
    checks['progressive_cross_beats_matched_last_both'] = all(
        p['conv_cross_vs_conv_last']['net'] > 0 for p in primary)
    checks['conditional_cross_beats_matched_last_both'] = all(
        p['conditional_cross_vs_conditional_last']['net'] > 0 for p in primary)
    return dict(gain_checks=checks, stable_gain_proven=False,
        scope=POLICY['scope'], verdict='MATCHED_EXPLORATORY_STUDY_COMPLETE',
        note='Completion is not success; report secondary320 and all negative results. Nominal p values are not multiplicity-corrected.')


def project_sources(root, initial):
    """Static local imports, without importing/executing model code."""
    root = Path(root)
    pending, found = list(initial), set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        if not name.endswith('.py'):
            continue
        nodes = ast.walk(ast.parse((root / name).read_text(encoding='utf-8-sig')))
        for node in nodes:
            modules = ([x.name for x in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and not node.level else [])
            for module in modules:
                if not module or module.split('.')[0] not in ('src', 'scripts', 'argparser'):
                    continue
                path = Path(*module.split('.'))
                for candidate in (path.with_suffix('.py'), path / '__init__.py'):
                    if (root / candidate).is_file():
                        pending.append(candidate.as_posix())
    return sorted(found)
