"""CPU mechanics and fixed-protocol regression; real-image smoke runs on GPU1."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from src.models.hcen_followup import ARMS, COUNTS, build_model, module_gradient_status
from src.models.hcen_ablation import build_model as original_model, common_parameters
from src.hcen_followup_protocol import POLICY, training_schedule, effective_policy, comparable_contract, summarize, aggregate_reports
from src.hcen_aux_ablation import training_schedule as original_schedule

ROOT = Path(__file__).resolve().parents[1]


class Visual(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.semantic_region_gate = None
        self.aggregator = nn.Sequential(nn.Flatten(), nn.Linear(channels*4*4, 16))


@pytest.mark.parametrize('arm', ARMS)
def test_registered_counts_and_no_sasf(arm):
    visual = nn.Module(); visual.semantic_region_gate = None; visual.aggregator = nn.Identity()
    model = build_model(visual, arm, 42)
    assert sum(p.numel() for p in model.neck.parameters()) == COUNTS[arm]
    assert not any('.spatial.' in name for name in model.state_dict())
    assert all(name.startswith('neck.') for name, p in model.named_parameters() if p.requires_grad)
    assert model.mode == ('last_only' if arm == 'last_only' else 'hierarchy')


@pytest.mark.parametrize('s', (42, 43, 44))
def test_scan_and_last_exact_same_parameters_and_input_is_only_difference(s):
    v = Visual(); maps = [torch.randn(2, 8, 4, 4) for _ in range(3)]
    scan = build_model(v, 'scan', s, channels=8, dim=8, inner=8, state=2)
    last = build_model(v, 'last_only', s, channels=8, dim=8, inner=8, state=2)
    assert all(torch.equal(p, last.neck.state_dict()[n]) for n, p in scan.neck.state_dict().items())
    with torch.no_grad():
        scan.neck.up.weight.normal_(0, .01); last.neck.load_state_dict(scan.neck.state_dict())
    assert torch.equal(last.aggregate(*maps)[0], scan.aggregate(maps[-1], maps[-1], maps[-1])[0])
    assert not torch.equal(last.aggregate(*maps)[0], scan.aggregate(*maps)[0])


@pytest.mark.parametrize('arm', ARMS)
def test_zero_start_and_actual_two_stage_gradients(arm):
    v = Visual(); maps = [torch.randn(4, 8, 4, 4) for _ in range(3)]
    model = build_model(v, arm, 42, channels=8, dim=8, inner=8, state=2)
    out = model.aggregate(*maps, auxiliary=True)
    assert torch.equal(out[0], v.aggregator(maps[-1])) and not out[-1].any()
    opt = torch.optim.AdamW(model.neck.parameters(), lr=.001, weight_decay=0.)
    seen = {'0': False, '1': False}
    for _ in range(4):
        opt.zero_grad(set_to_none=True); out = model.aggregate(*maps, auxiliary=True)
        loss = out[0].square().mean()+sum(x.square().mean()*0 for x in out[1:3]); loss.backward()
        for k, yes in module_gradient_status(model).items(): seen[k] |= yes
        assert all(p.grad is None for p in v.parameters())
        opt.step()
        ratio = out[-1].flatten(1).norm(dim=1)/maps[-1].flatten(1).norm(dim=1)
        assert (ratio <= .100001).all()
    assert all(seen.values())


def test_factory_unchanged_shared_weights_and_zero_mixer_outputs():
    v = Visual(); options = dict(channels=8, dim=8, inner=8, state=2)
    old = original_model(v, 'no_sasf', 43, **options)
    scan = build_model(v, 'scan', 43, **options)
    conv = build_model(v, 'conv', 43, **options)
    assert all(torch.equal(p, scan.neck.state_dict()[n]) for n, p in old.neck.state_dict().items())
    for arm in ARMS:
        m = build_model(v, arm, 43, **options)
        assert all(torch.equal(p, common_parameters(m)[n]) for n, p in common_parameters(scan).items())
    low = torch.randn(2, 8, 4, 4); high = torch.randn_like(low)
    for a, b in zip(scan.neck.stages, conv.neck.stages):
        assert torch.equal(a(low, high), b(low, high))
        assert torch.equal(a.ssm.in_proj.weight, b.conv_mixer.in_proj.weight)
    assert COUNTS['conv']-COUNTS['scan'] == 1088


def test_real_sealed_initialization_if_present():
    from scripts.train_slgd import parameter_hash, read
    base = ROOT/'logs/hcen_module_ablation_v1'
    if not (base/'seed42/no_sasf_screen/initial.json').exists():
        pytest.skip('Sealed reference unavailable')
    for s in (42, 43, 44):
        for arm, old in [('scan', 'no_sasf'), ('last_only', 'no_sasf'), ('linear', 'linear')]:
            v = nn.Module(); v.semantic_region_gate = None; v.aggregator = nn.Identity()
            assert parameter_hash(build_model(v, arm, s)) == read(base/('seed'+str(s))/(old+'_screen')/'initial.json')['parameter_sha256']


def triplets(n=8192):
    anchor = np.arange(n)*4
    return np.stack((anchor, anchor+1, np.roll(anchor, 1)), 1)


def test_exact_four_passes_and_smoke_prefix_no_aliases():
    rows = triplets(); old = original_schedule(rows, 'screen')
    long = training_schedule(rows, 'screen')
    assert len(long) == 2048
    assert all(long[start:start+512] == old for start in (0, 512, 1024, 1536))
    assert training_schedule(rows, 'smoke') == long[:8]
    assert long[0] is not long[512]
    assert POLICY['screen_steps'] == 2048 and POLICY['auxiliary_weight'] == POLICY['rank_weight'] == 0
    with pytest.raises(ValueError): training_schedule(rows, 'invalid')
    with pytest.raises(ValueError): training_schedule(triplets(12), 'screen')


def test_contract_only_depth_arm_and_seed_may_differ():
    keys = ('stage', 'checkpoint_sha256', 'source_completed_sha256', 'cache_completed_sha256',
        'metadata', 'train_places', 'original_code', 'benchmark_metadata', 'versions', 'schedule',
        'baseline_completed_sha256', 'followup_policy', 'reference_evidence', 'code')
    c = {k: {} for k in keys}; c.update(policy=effective_policy(42), initialization_seed=42, arm='scan', mode='hierarchy')
    other = copy.deepcopy(c); other.update(policy=effective_policy(43), initialization_seed=43, arm='last_only', mode='last_only')
    assert comparable_contract(c) == comparable_contract(other)
    other['schedule'] = [[12]]
    assert comparable_contract(c) != comparable_contract(other)
    other['initialization_seed'] = 44
    with pytest.raises(ValueError): comparable_contract(other)


def test_paired_reports_and_all_seed_aggregation():
    values = {arm: [True, False, False] for arm in ARMS}; values['scan'] = [True, True, False]
    p = summarize(values, [True, False, False], [0, 0, 1], 1)
    assert p['all']['comparisons']['scan_vs_conv'] == dict(corrections=[1], regressions=[], net=1)
    assert len(p['all']['comparisons']) == 3
    with pytest.raises(ValueError): summarize({'scan': [True]}, [True], [0], 1)
    with pytest.raises(ValueError): summarize(values, [True], [0], 1)
    reports = {s: {'gsv_'+k: copy.deepcopy(v) for k, v in p.items()} for s in (42, 43, 44)}
    for s in reports: reports[s].update({'msls-val': copy.deepcopy(p['all']), 'pitts30k-val': copy.deepcopy(p['all'])})
    result = aggregate_reports(reports)
    assert result['metrics']['pitts30k-val']['comparisons']['scan_vs_conv']['net_by_seed'] == {42: 1, 43: 1, 44: 1}
    assert all(result['directional_checks'].values())
    reports[44]['msls-val']['variants']['scan']['correct'] = 0
    assert not aggregate_reports(reports)['directional_checks']['msls_no_regression_vs_plain_every_seed']
    reports.pop(44)
    with pytest.raises(ValueError): aggregate_reports(reports)
    empty = summarize({a: [True] for a in ARMS}, [True], [0], 1)
    assert empty['evaluation']['variants']['scan']['r1'] is None
    json.dumps(empty, allow_nan=False)


def test_safe_launcher_fixed_protocol_and_no_old_source_edits():
    launcher = (ROOT/'scripts/run_hcen_followup.sh').read_bytes()
    assert b'\r' not in launcher
    text = launcher.decode()
    assert 'CUDA_VISIBLE_DEVICES=1' in text and 'flock -n 9' in text
    assert 'for ARM in scan linear conv last_only' in text
    assert text.index('hcen_followup.py wait_gpu') < text.index('hcen_followup.py train')
    assert 'for SEED in 42 43 44; do train_stage "$SEED" smoke; done' in text and 'SMOKE_PASS' in text
    helper = (ROOT/'scripts/hcen_followup.py').read_text(encoding='utf-8')
    assert 'save(args.output/\'last.pt\'' in helper and 'restore(args.output/\'last.pt\'' in helper
    assert 'os.kill(' not in helper and 'gpu-reset' not in helper
    assert 'build_model(visual, args.arm, args.seed)' in helper
    old = ROOT/'scripts/hcen_plain_multiseed.py'
    assert hashlib.sha256(old.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == 'f6801c13587cda6e8924cbc03190e1d29fe33a8beadc094de36d0b1cbcb4d7ea'
