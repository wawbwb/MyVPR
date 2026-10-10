"""Matched GSV-only SLGD teacher/A/B screen; no CLIP or benchmark tuning."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.slgd import (LocalProjector, SLGDStudent, local_tokens, mine_pairs,
                             mutual_score, local_objective, compression_loss)

RU_SHA = '38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e'
POLICY = dict(revision=1, seed=42, train_places=8192, holdout_places=1024,
              views=4, places_per_batch=16, size=280, microbatch=4,
              smoke_steps=4, pilot_steps=512, beta=.2, slots=16,
              local_dim=128, local_grid=10, local_margin=.05,
              teacher_lr=1e-4, local_lr=1e-4, boq_lr=1e-5,
              local_weight=.1, compression_weight=1., temperature=.07,
              precision='fp32', clip=1., weight_decay=0.,
              teacher_selection='Fixed last; frozen before either student starts',
              teacher_gate=dict(min_valid_fraction=.95, min_pair_accuracy=.65,
                                min_gain_vs_raw_pp=1., min_mean_margin=0.),
              student_gate=dict(min_correct_gain=5, min_local_gain_pp=1.),
              scope='Exploratory single-seed GSV place-heldout feasibility; not independent final validation',
              inference='One RU-global + OT-slot vector, one dot product; no pair matching/CLIP',
              limitations='Pooled native tokens; MNN are pseudo correspondences; GSV different places may overlap')
SOURCES = ('scripts/train_slgd.py', 'src/models/slgd.py', 'scripts/check_slgd_gate.py',
           'src/models/aggregators/salad.py', 'src/dataloaders/train/gsv_cities.py',
           'scripts/eval_condition_robustness.py', 'src/models/backbones/dinov2.py',
           'src/models/aggregators/boq.py', 'src/losses/vpr_losses.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def verify(path):
    path = Path(path).resolve()
    completed = read(path/'completed.json')
    if not completed['complete']:
        raise ValueError('Incomplete run')
    for name, digest in completed['files'].items():
        target = (path/name).resolve()
        if not target.is_relative_to(path) or sha(target) != digest:
            raise ValueError('Output hash changed: '+name)
    return read(path/'contract.json')


def seal(path):
    write(path/'completed.json', dict(complete=True, files={p.name: sha(p)
          for p in path.iterdir() if p.is_file() and p.name != 'completed.json' and p.suffix != '.tmp'}))


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class MatchedPlaces(Dataset):
    def __init__(self, dataset, indices, epoch=0):
        self.dataset, self.indices, self.epoch = dataset, indices, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            value = int(hashlib.sha256(f'slgd42:{self.epoch}:{self.dataset.places_ids[i]}'.encode()).hexdigest()[:8], 16)
            random.seed(value)
            np.random.seed(value)
            torch.random.default_generator.manual_seed(value)
            return self.dataset[i]
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save(path, model, optimizer, state):
    payload = dict(parameters={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   optimizer=optimizer.state_dict(), state=state,
                   rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore(path, model, optimizer, contract_sha):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(active) != set(payload['parameters']) or payload['state']['contract_sha256'] != contract_sha:
        raise ValueError('Checkpoint parameter/contract identity differs')
    with torch.no_grad():
        for name, parameter in active.items():
            parameter.copy_(payload['parameters'][name].to(parameter.device))
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['rng'])
    torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['state']


def teacher_checks(report):
    limits = POLICY['teacher_gate']
    return dict(valid_fraction=report['valid_fraction'] >= limits['min_valid_fraction'],
                accuracy=report['pair_accuracy'] >= limits['min_pair_accuracy'],
                raw_gain=(report['pair_accuracy']-report['raw_pair_accuracy'])*100 >= limits['min_gain_vs_raw_pp'],
                margin=report['mean_margin'] > limits['min_mean_margin'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('teacher', 'retrieval', 'local_distill'), required=True)
    parser.add_argument('--stage', choices=('smoke', 'pilot'), default='pilot')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    import fcntl
    from torchvision import transforms as T
    from tqdm import tqdm
    from src.dataloaders.train.gsv_cities import GSVCitiesDataset
    from src.losses.vpr_losses import VPRLossFunction
    from scripts.eval_condition_robustness import load_inference_model_from_ckpt

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU1 only: CUDA_VISIBLE_DEVICES=1')
    if not args.checkpoint.is_file() or sha(args.checkpoint) != RU_SHA:
        raise ValueError('Missing original RU or SHA256 mismatch')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.65, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.output.parent/(args.output.name+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seed(42)
    begin = [T.Resize((280, 280), interpolation=T.InterpolationMode.BICUBIC)]
    end = [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])]
    augment = T.Compose(begin+[T.ColorJitter(.4, .4, .4, .1), T.RandomGrayscale(.2)]+end)
    clean = T.Compose(begin+end)
    cities = sorted(p.stem for p in (args.dataset_root/'Dataframes').glob('*.csv'))
    if not cities:
        raise ValueError('GSV metadata missing')
    train = GSVCitiesDataset(dataset_path=args.dataset_root, cities=cities, img_per_place=4, transform=augment)
    # GSVCitiesDataset shuffles each city dataframe during construction.
    # Reset before the second construction, not only before per-place sampling.
    seed(42)
    dev = GSVCitiesDataset(dataset_path=args.dataset_root, cities=cities, img_per_place=4, transform=clean)
    if list(train.places_ids) != list(dev.places_ids):
        raise ValueError('Dataset ordering differs')
    order = sorted(range(len(train)), key=lambda i: hashlib.sha256(f'slgd-partition42:{train.places_ids[i]}'.encode()).hexdigest())
    if len(order) < POLICY['train_places']+POLICY['holdout_places']:
        raise ValueError('Insufficient places')
    holdout = order[:POLICY['holdout_places']]
    training = order[POLICY['holdout_places']:POLICY['holdout_places']+POLICY['train_places']]
    schedule = np.random.default_rng(74031).permutation(training).tolist()
    steps = POLICY[args.stage+'_steps']
    schedule = schedule[:steps*16]
    holdout = holdout[:32] if args.stage == 'smoke' else holdout
    contract = dict(mode=args.mode, stage=args.stage, policy=POLICY, checkpoint_sha256=RU_SHA,
                    train_places=[str(train.places_ids[i]) for i in schedule],
                    holdout_places=[str(train.places_ids[i]) for i in holdout],
                    metadata={p.name: sha(p) for p in sorted((args.dataset_root/'Dataframes').glob('*.csv'))},
                    code={name: code_hash(ROOT/name) for name in SOURCES},
                    versions=dict(torch=torch.__version__, numpy=np.__version__), teacher_sha256=None)
    if args.mode != 'teacher':
        if args.teacher is None:
            parser.error('Student requires --teacher completed run')
        teacher_contract = verify(args.teacher)
        for name in ('stage', 'policy', 'checkpoint_sha256', 'train_places', 'holdout_places', 'metadata', 'code', 'versions'):
            if teacher_contract[name] != contract[name]:
                raise ValueError('Teacher/student contract differs: '+name)
        if args.stage != 'smoke' and not all(teacher_checks(read(args.teacher/'final.json')).values()):
            raise ValueError('Teacher efficacy gate failed; inspect report before any student training')
        contract['teacher_sha256'] = sha(args.teacher/'completed.json')
    if args.output.exists():
        if not args.resume or read(args.output/'contract.json') != contract:
            raise ValueError('Existing output or immutable contract mismatch')
        if (args.output/'completed.json').exists():
            verify(args.output)
            print('Already complete:', args.output, flush=True)
            return
    else:
        args.output.mkdir()
    write(args.output/'contract.json', contract)
    visual = load_inference_model_from_ckpt(args.checkpoint, 'cpu').cuda().eval().requires_grad_(False)
    frozen_aggregator = copy.deepcopy(visual.aggregator).eval().requires_grad_(False)
    seed(42)
    teacher = LocalProjector().cuda()
    if args.mode != 'teacher':
        parameters = torch.load(args.teacher/'last.pt', map_location='cpu', weights_only=True)['parameters']
        teacher.load_state_dict(parameters, strict=True)
        teacher.requires_grad_(False).eval()
        seed(42)
        model = SLGDStudent(visual, teacher, slots=POLICY['slots'], beta=POLICY['beta']).cuda().eval()
        global_params = list(model.visual.aggregator.parameters())
        global_ids = {id(p) for p in global_params}
        local_params = [p for p in model.parameters() if p.requires_grad and id(p) not in global_ids]
        optimizer = torch.optim.AdamW([dict(params=global_params, lr=POLICY['boq_lr']),
                                      dict(params=local_params, lr=POLICY['local_lr'])], weight_decay=0.)
    else:
        model = teacher
        optimizer = torch.optim.AdamW(model.parameters(), lr=POLICY['teacher_lr'], weight_decay=0.)
    active = [p for p in model.parameters() if p.requires_grad]
    state = dict(steps=0, rows=[], contract_sha256=sha(args.output/'contract.json'), initial_parameter_sha256=parameter_hash(model))
    if (args.output/'last.pt').is_file():
        state = restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    loss_fn = VPRLossFunction()

    @torch.no_grad()
    def extract(images):
        features = []
        for chunk in images.cuda().split(POLICY['microbatch']):
            f = visual.backbone(chunk)
            features.append(f[0] if isinstance(f, tuple) else f)
        return torch.cat(features)

    @torch.no_grad()
    def frozen_global(features):
        gated = features
        if visual.semantic_region_gate is not None:
            gated = visual.semantic_region_gate(gated)[0]
        result = frozen_aggregator(gated)
        return result[0] if isinstance(result, tuple) else result

    @torch.no_grad()
    def evaluate():
        model.eval()
        globals_, locals_, raw_, descriptors, student_locals = [], [], [], [], []
        data = MatchedPlaces(dev, holdout, epoch=99)
        write(args.output/'progress.json', dict(phase='development', steps=state['steps'], places=len(data)))
        for images, _ in tqdm(DataLoader(data, batch_size=4, num_workers=args.workers), desc=args.mode+' GSV dev'):
            f = extract(images.flatten(0, 1))
            globals_.append(frozen_global(f).cpu())
            raw_.append(torch.nn.functional.normalize(local_tokens(f), dim=-1).cpu())
            locals_.append((model(local_tokens(f)) if args.mode == 'teacher' else teacher(local_tokens(f))).cpu())
            if args.mode != 'teacher':
                results, local = model.aggregate(f)
                descriptors.append(results.cpu())
                student_locals.append(local.cpu())
        globals_ = torch.cat(globals_).cuda()
        raw_ = torch.cat(raw_)
        locals_ = torch.cat(locals_)
        query = torch.arange(0, len(globals_), 4, device='cuda')
        gallery = torch.arange(len(globals_), device='cuda')
        gallery = gallery[gallery % 4 != 0]
        scores = globals_[query] @ globals_[gallery].T
        same = query[:, None]//4 == gallery[None]//4
        pos = gallery[scores.masked_fill(~same, -torch.inf).argmax(1)].cpu()
        neg = gallery[scores.masked_fill(same, -torch.inf).argmax(1)].cpu()
        q = query.cpu()
        def match_margins(tokens):
            margins, valid = [], []
            for start in range(0, len(q), 32):
                qi, pi, ni = q[start:start+32], pos[start:start+32], neg[start:start+32]
                ps, pc = mutual_score(tokens[qi].cuda(), tokens[pi].cuda())
                ns, nc = mutual_score(tokens[qi].cuda(), tokens[ni].cuda())
                margins.extend((ps-ns).cpu().tolist())
                valid.extend(((pc >= 3) & (nc >= 3)).cpu().tolist())
            return np.asarray(margins), np.asarray(valid)
        margins, valid = match_margins(locals_)
        raw_margins, raw_valid = match_margins(raw_)
        report = dict(queries=len(q), gallery=len(gallery), query_indices=q.tolist(),
                      positive_indices=pos.tolist(), negative_indices=neg.tolist(),
                      ru_correct=int((gallery[scores.argmax(1)]//4 == query//4).sum()),
                      pair_accuracy=float((margins > 0).mean()), raw_pair_accuracy=float((raw_margins > 0).mean()),
                      valid_fraction=float(valid.mean()), raw_valid_fraction=float(raw_valid.mean()),
                      mean_margin=float(margins.mean()), margins=margins.tolist(), valid=valid.tolist(),
                      positive_choice='Best frozen-RU same-place view (does not certify spatial overlap)',
                      negative_choice='Hardest frozen-RU other-place image in 3072-image heldout gallery')
        if args.mode != 'teacher':
            d = torch.cat(descriptors).cuda()
            student_scores = d[query] @ d[gallery].T
            pred = gallery[student_scores.argmax(1)]
            outcomes = (pred//4 == query//4).cpu().tolist()
            report.update(correct=sum(outcomes), outcomes=outcomes, predictions=pred.cpu().tolist(),
                          descriptor_dimension=d.shape[1], descriptor_norm_max_error=float((d.norm(dim=1)-1).abs().max()))
            student_margin, student_valid = match_margins(torch.cat(student_locals))
            report.update(student_local_pair_accuracy=float((student_margin > 0).mean()),
                          student_local_mean_margin=float(student_margin.mean()),
                          student_local_valid_fraction=float(student_valid.mean()))
        print({k: v for k, v in report.items() if k not in ('query_indices','positive_indices','negative_indices','margins','valid','outcomes','predictions')}, flush=True)
        return report

    if not (args.output/'initial.json').is_file():
        write(args.output/'initial.json', evaluate())
        save(args.output/'last.pt', model, optimizer, state)
    elif state['steps'] == 0:
        # Initial report must be paired with the checkpoint saved before training.
        initial = read(args.output/'initial.json')
        if not initial['queries']:
            raise ValueError('Invalid initial report')
    model.train()
    remaining = schedule[state['steps']*16:]
    bar = tqdm(DataLoader(MatchedPlaces(train, remaining), batch_size=16, num_workers=args.workers, drop_last=True),
               total=steps, initial=state['steps'], desc=args.mode+' '+args.stage)
    for images, labels in bar:
        optimizer.zero_grad(set_to_none=True)
        f = extract(images.flatten(0, 1))
        frozen = frozen_global(f)
        pos, neg = mine_pairs(frozen, labels.flatten().cuda())
        if args.mode == 'teacher':
            loss, margin, valid = local_objective(model(local_tokens(f)), pos, neg, POLICY['local_margin'])
            vpr = loss*0
            local_loss = loss
            distill = loss*0
            reliable = valid & (margin.detach() > 0)
        else:
            descriptor, tokens = model.aggregate(f)
            vpr, _ = loss_fn(descriptor, labels.flatten().cuda())
            local_loss, _, _ = local_objective(tokens, pos, neg, POLICY['local_margin'])
            with torch.no_grad():
                _, margin, valid = local_objective(teacher(local_tokens(f)), pos, neg, POLICY['local_margin'])
                reliable = valid & (margin > 0)
            distill = compression_loss(descriptor, pos, neg, margin, reliable, POLICY['temperature'])
            loss = vpr if args.mode == 'retrieval' else vpr+POLICY['local_weight']*local_loss+POLICY['compression_weight']*distill
        if not torch.isfinite(loss):
            raise ValueError('Non-finite SLGD loss')
        gradient_probe = None
        if args.mode == 'local_distill' and state['steps'] % 64 == 0:
            params = list(model.local.parameters())+list(model.slot_score.parameters())+[model.dustbin]
            gv = torch.autograd.grad(vpr, params, retain_graph=True, allow_unused=True)
            auxiliary = POLICY['local_weight']*local_loss+POLICY['compression_weight']*distill
            ga = torch.autograd.grad(auxiliary, params, retain_graph=True, allow_unused=True)
            v2 = sum(float(g.detach().double().square().sum()) for g in gv if g is not None)
            a2 = sum(float(g.detach().double().square().sum()) for g in ga if g is not None)
            dot = sum(float((x.detach().double()*y.detach().double()).sum())
                      for x, y in zip(gv, ga) if x is not None and y is not None)
            gradient_probe = dict(vpr_norm=v2**.5, auxiliary_norm=a2**.5,
                                  ratio=(a2/v2)**.5 if v2 > 0 else None,
                                  cosine=dot/(v2*a2)**.5 if v2*a2 > 0 else None)
        loss.backward()
        # Identical optimizer clocks even for temporarily unused parameters.
        for parameter in active:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        norm = torch.nn.utils.clip_grad_norm_(active, POLICY['clip'], error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.backbone.parameters()):
            raise ValueError('Frozen backbone received gradients')
        optimizer.step()
        state['steps'] += 1
        row = dict(step=state['steps'], loss=float(loss.detach()), vpr=float(vpr.detach()),
                   local=float(local_loss.detach()), compression=float(distill.detach()),
                   gradient_norm=float(norm), reliable_fraction=float(reliable.float().mean()))
        if gradient_probe is not None:
            row['local_branch_gradient_probe'] = gradient_probe
        state['rows'].append(row)
        bar.set_postfix(loss=row['loss'], kd=row['compression'], coverage=row['reliable_fraction'])
        write(args.output/'progress.json', dict(phase='training', mode=args.mode, steps=state['steps'], total=steps))
        if state['steps'] % 16 == 0:
            save(args.output/'last.pt', model, optimizer, state)
    if state['steps'] != steps:
        raise ValueError('Training schedule was not completed')
    final = evaluate()
    write(args.output/'final.json', final)
    save(args.output/'last.pt', model, optimizer, state)
    before = parameter_hash(model)
    restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    if parameter_hash(model) != before:
        raise ValueError('Checkpoint roundtrip differs')
    clocks = [int(optimizer.state[p]['step']) for p in active]
    if any(clock != steps for clock in clocks):
        raise ValueError('Optimizer clock mismatch')
    summary = dict(mode=args.mode, stage=args.stage, steps=steps,
                   initial_parameter_sha256=state['initial_parameter_sha256'], final_parameter_sha256=before,
                   parameters_changed=before != state['initial_parameter_sha256'], checkpoint_roundtrip=True,
                   trainable_parameters=sum(p.numel() for p in active),
                   nonzero_gradient_steps=sum(r['gradient_norm'] > 0 for r in state['rows']),
                   mean_reliable_fraction=float(np.mean([r['reliable_fraction'] for r in state['rows']])),
                   rows=state['rows'], teacher_checks=teacher_checks(final),
                   status='TRAINING COMPLETE; not a stable efficacy/novelty verdict')
    write(args.output/'summary.json', summary)
    write(args.output/'progress.json', dict(phase='complete', steps=steps))
    seal(args.output)
    print('COMPLETE:', args.output, flush=True)


if __name__ == '__main__':
    main()
