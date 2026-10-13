"""Scientific identity, actual module learning and adapted-paper equations."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from src.models.conv_fusion_controls import ARMS, COUNTS, TokenMixer, build_model, interface_parameters, gradient_inventory, module_gradient_status
from src.conv_fusion_protocol import POLICY, REFERENCE_SHA, training_schedule, matched_contract, summarize, aggregate_reports

ROOT = Path(__file__).resolve().parents[1]


class Visual(nn.Module):
    def __init__(self, channels=8):
        super().__init__(); self.semantic_region_gate = None
        self.aggregator = nn.Sequential(nn.Flatten(), nn.Linear(channels*4*4, 16))


@pytest.mark.parametrize('arm', ('conv', *ARMS))
def test_counts_no_unused_removed_modules(arm):
    v = nn.Module(); v.semantic_region_gate = None; v.aggregator = nn.Identity()
    model = build_model(v, arm, 42)
    assert sum(p.numel() for p in model.neck.parameters()) == COUNTS[arm]
    assert all(n.startswith('neck.') for n, p in model.named_parameters() if p.requires_grad)
    if arm == 'concat_token':
        assert not hasattr(model.neck, 'stages') and not hasattr(model.neck, 'projections')
        assert not any('.spatial.' in n or '.ssm.' in n for n in model.state_dict())
        assert len(model.neck.tokenmix) == 2
    assert model.mode == ('last_only' if arm == 'conv_last' else 'hierarchy')


@pytest.mark.parametrize('s', (42, 43, 44))
def test_conv_last_exact_same_weights_only_input_differs(s):
    v = Visual(); options = dict(channels=8, dim=8, inner=8, state=2, grid=(4, 4))
    conv = build_model(v, 'conv', s, **options); last = build_model(v, 'conv_last', s, **options)
    assert all(torch.equal(p, last.neck.state_dict()[n]) for n, p in conv.neck.state_dict().items())
    maps = [torch.randn(2, 8, 4, 4) for _ in range(3)]
    with torch.no_grad():
        conv.neck.up.weight.normal_(0, .01); last.neck.load_state_dict(conv.neck.state_dict())
    assert torch.equal(last.aggregate(*maps)[0], conv.aggregate(maps[-1], maps[-1], maps[-1])[0])
    assert not torch.equal(last.aggregate(*maps)[0], conv.aggregate(*maps)[0])


def test_interface_initialization_preserved_and_paper_formula():
    v = Visual(); options = dict(channels=8, dim=8, inner=8, state=2, grid=(4, 4))
    original = build_model(v, 'conv', 43, **options)
    for arm in ARMS:
        m = build_model(v, arm, 43, **options)
        assert all(torch.equal(p, interface_parameters(m)[n]) for n, p in interface_parameters(original).items())
    mixer = TokenMixer(8, 16); x = torch.randn(2, 8, 16)
    official_equation = x + mixer.mix(mixer.norm(x.permute(0, 2, 1)).permute(0, 2, 1))
    assert torch.equal(mixer(x), official_equation)
    m = build_model(v, 'concat_token', 43, **options)
    maps = [torch.randn(2, 8, 4, 4) for _ in range(3)]
    with pytest.raises(ValueError): m.aggregate(*(f[:, :, :3] for f in maps))


@pytest.mark.parametrize('arm', ARMS)
def test_strict_zero_start_bounded_updates_and_actual_module_gradients(arm):
    v = Visual(); model = build_model(v, arm, 42, channels=8, dim=8, inner=8, state=2, grid=(4, 4))
    maps = [torch.randn(4, 8, 4, 4) for _ in range(3)]
    out = model.aggregate(*maps, auxiliary=True)
    assert torch.equal(out[0], v.aggregator(maps[-1])) and not out[-1].any()
    initial_heads = {n: p.detach().clone() for n, p in model.neck.aux.named_parameters()}
    opt = torch.optim.AdamW(model.neck.parameters(), lr=.001, weight_decay=0.)
    seen = {k: False for k in gradient_inventory(arm)}
    for _ in range(4):
        opt.zero_grad(set_to_none=True); out = model.aggregate(*maps, auxiliary=True)
        (out[0].square().mean()+sum(x.square().mean()*0 for x in out[1:3])).backward()
        for k, yes in module_gradient_status(model).items(): seen[k] |= yes
        assert all(p.grad is None for p in v.parameters())
        opt.step()
        ratio = out[-1].flatten(1).norm(dim=1)/maps[-1].flatten(1).norm(dim=1)
        assert (ratio <= .100001).all()
    assert all(seen.values())
    assert all(torch.equal(p, initial_heads[n]) for n, p in model.neck.aux.named_parameters())


def test_real_sealed_initialization_and_reference_contract_if_present():
    from scripts.train_slgd import parameter_hash, read
    from scripts.conv_fusion_controls import references
    base = ROOT/'logs/hcen_simplified_2048_v1'
    if not (base/'aggregate/completed.json').exists(): pytest.skip('Sealed reference unavailable')
    args = SimpleNamespace(reference=ROOT/'logs/hcen_plain_base_v1', full_reference=ROOT/'logs/hcen_plain_multiseed_v1',
        module_reference=ROOT/'logs/hcen_module_ablation_v1', followup_reference=base)
    _, _, identity = references(args)
    assert identity['aggregate_sha256'] == REFERENCE_SHA
    for s in (42, 43, 44):
        v = nn.Module(); v.semantic_region_gate = None; v.aggregator = nn.Identity()
        old = read(base/('seed'+str(s))/'conv_screen/initial.json')
        assert parameter_hash(build_model(v, 'conv_last', s)) == old['parameter_sha256']
        c = read(base/('seed'+str(s))/'conv_screen/contract.json'); new = copy.deepcopy(c)
        new.update(arm='conv_last', mode='last_only', code={'new': 'implementation'}, conv_control_policy=POLICY)
        assert matched_contract(new) == matched_contract(c)
        new['schedule'][0] = [999]
        assert matched_contract(new) != matched_contract(c)


def test_same_2048_budget_and_paired_aggregation_not_best_seed():
    a = np.arange(8192)*4; rows = np.stack((a, a+1, np.roll(a, 1)), 1)
    schedule = training_schedule(rows, 'screen')
    assert len(schedule) == 2048 and training_schedule(rows, 'smoke') == schedule[:8]
    assert schedule[:512] == schedule[512:1024] == schedule[1024:1536] == schedule[1536:]
    values = {arm: [True, False, False] for arm in ('conv', *ARMS)}; values['conv'] = [True, True, False]
    p = summarize(values, [True, False, False], [0, 0, 1], 1)
    assert p['all']['comparisons']['conv_vs_conv_last'] == dict(corrections=[1], regressions=[], net=1)
    reports = {s: {'gsv_'+k: copy.deepcopy(v) for k, v in p.items()} for s in (42, 43, 44)}
    for s in reports: reports[s].update({'msls-val': copy.deepcopy(p['all']), 'pitts30k-val': copy.deepcopy(p['all'])})
    r = aggregate_reports(reports)
    assert r['metrics']['msls-val']['variants']['conv']['mean_r1'] == pytest.approx(2/3)
    assert all(r['directional_checks'].values())
    reports[44]['msls-val']['comparisons']['conv_vs_conv_last']['net'] = -1
    assert not aggregate_reports(reports)['directional_checks']['msls-val_conv_vs_conv_last']
    reports.pop(44)
    with pytest.raises(ValueError): aggregate_reports(reports)
    with pytest.raises(ValueError): summarize({'conv': [True]}, [True], [0], 1)
    with pytest.raises(ValueError): summarize(values, [True], [0], 1)
    empty = summarize({a: [True] for a in values}, [True], [0], 1)
    assert empty['evaluation']['variants']['conv']['r1'] is None
    json.dumps(empty, allow_nan=False)


def test_fixed_safe_queue_no_old_source_edits():
    raw = (ROOT/'scripts/run_conv_fusion_controls.sh').read_bytes(); text = raw.decode()
    assert b'\r' not in raw and 'CUDA_VISIBLE_DEVICES=1' in text and 'flock -n 9' in text
    assert 'for ARM in conv_last concat_token' in text and 'SMOKE_PASS' in text
    assert 'for SEED in 42 43 44; do train_stage "$SEED" smoke; done' in text
    assert text.index('conv_fusion_controls.py wait_gpu') < text.index('conv_fusion_controls.py train')
    helper = (ROOT/'scripts/conv_fusion_controls.py').read_text(encoding='utf-8')
    assert 'os.kill(' not in helper and 'gpu-reset' not in helper
    assert "save(args.output/'last.pt'" in helper and "restore(args.output/'last.pt'" in helper
    assert 'conv_last exactly same parameters' in POLICY['fairness']
    assert 'not complete reproduction' in POLICY['paper']
    from scripts.train_slgd import code_hash
    base = ROOT/'logs/hcen_simplified_2048_v1/seed42/conv_screen/contract.json'
    if base.exists():
        sealed = json.loads(base.read_text())
        assert all(code_hash(ROOT/p) == digest for p, digest in sealed['code'].items())
