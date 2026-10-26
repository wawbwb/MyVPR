"""Paired six-cell effects: package, scale-only, gate-only, and interaction."""
import numpy as np
from src.scale_repeat_statistics import contrast, paired_interval

CELLS = dict(A=('official', 'base_322'), B=('official', 'descriptor_average'), C=('official', 'group_gate'),
    D=('innovation1', 'base_322'), E=('innovation1', 'descriptor_average'), F=('innovation1', 'group_gate'))
COMPARISONS = dict(C_minus_A=('C', 'A'), B_minus_A=('B', 'A'), C_minus_B=('C', 'B'),
    F_minus_D=('F', 'D'), E_minus_D=('E', 'D'), F_minus_E=('F', 'E'),
    D_minus_A=('D', 'A'), E_minus_B=('E', 'B'), F_minus_C=('F', 'C'))


def interaction(delta, clusters, repetitions, seed):
    delta = np.asarray(delta, float)
    if delta.ndim != 2 or delta.shape[0] < 2:
        raise ValueError('Matched seed-by-query effects required')
    counts = delta.sum(1)
    return dict(seed_net_queries=counts.astype(int).tolist(), mean_net_queries=float(counts.mean()),
        query_paired=paired_interval(delta.mean(0), np.arange(delta.shape[1]), repetitions, seed),
        positive_component_paired=paired_interval(delta.mean(0), clusters, repetitions, seed))


def factorial(official, innovation1, clusters, repetitions=2000, seed=20261026):
    if len(official) != len(innovation1) or len(official) < 2:
        raise ValueError('Matched family seed counts required')
    families = dict(official=official, innovation1=innovation1)
    matrices = {k: np.asarray([r[name]['outcomes'] for r in families[family]], bool) for k, (family, name) in CELLS.items()}
    if len({v.shape for v in matrices.values()}) != 1 or any(v.ndim != 2 or v.shape[1] != len(clusters) for v in matrices.values()):
        raise ValueError('Matched family/seed/query matrices required')
    effects = {name: contrast(matrices[left], matrices[right], clusters, repetitions, seed)
        for name, (left, right) in COMPARISONS.items()}
    values = {k: v.astype(int) for k, v in matrices.items()}
    return dict(cell_correct={k: v.sum(1).tolist() for k, v in matrices.items()}, contrasts=effects,
        package_interaction=interaction(values['F']-values['D']-values['C']+values['A'], clusters, repetitions, seed),
        gate_interaction=interaction(values['F']-values['E']-values['C']+values['B'], clusters, repetitions, seed),
        note='Same queries across seeds/families are paired, never treated as independent replicas. Interaction does not establish causal architecture-only effects.')
