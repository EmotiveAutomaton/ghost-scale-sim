"""Independent portfolio reconstruction; no producer reader, selector or fitter."""
from collections import defaultdict
from itertools import product
import gzip
import json
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import read, write, file_digest, canonical
from ..v18_4.neural_data import TRAIN_QUERIES, NEW_QUERIES, FAR_QUERIES


def close(a, b, tolerance=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid reconstruction shape/value')
    error = float(np.max(np.abs(a - b))) if a.size else 0.
    if error > tolerance:
        raise ValueError(f'independent portfolio mismatch: {error}')
    return error


def repair(raw):
    r = np.asarray(raw).reshape(len(raw), -1, 16)
    bad = (r.min(2) < 0) | (r.max(2) > 1) | (np.abs(r.sum(2) - 1) > 1e-6)
    p = np.clip(r, 0, None) + 1e-6
    p /= p.sum(2, keepdims=True)
    return p.reshape(raw.shape), bad


def score(y, p):
    y, p = y.reshape(len(y), -1, 16), p.reshape(len(p), -1, 16)
    return -(y * np.log(np.maximum(p, 1e-300))).sum(2), ((y - p) ** 2).sum(2)


def ridge(x, y, penalty):
    # Center first: solve only the penalized slopes, recover intercept afterward.
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    weights = np.linalg.solve(xc.T @ xc + penalty * np.eye(x.shape[1]), xc.T @ yc)
    return np.vstack((ym - xm @ weights, weights))


def forecast(x, weights):
    return repair(weights[0] + x @ weights[1:])


def cols(indices):
    return np.concatenate([np.arange(j * 16, (j + 1) * 16) for j in indices])


def selections(teacher, learned, target, selection, selection_target, seed, size=5):
    count = teacher.shape[1] // 16
    chosen = dict(original=list(range(size)), random=sorted(
        np.random.default_rng(seed + 700).choice(count, size, replace=False).tolist()))
    diverse = [0]
    while len(diverse) < size:
        distances = {j: min(float(np.square(teacher[:, cols([j])] -
            teacher[:, cols([k])]).mean()) for k in diverse)
            for j in range(count) if j not in diverse}
        diverse.append(max(distances, key=distances.get))
    chosen['diversity'] = diverse
    targeted, search = [], []
    while len(targeted) < size:
        losses = {}
        for j in range(count):
            if j in targeted:
                continue
            c = cols(targeted + [j])
            head = ridge(learned[:, c], target, .01)
            p, _ = forecast(selection[:, c], head)
            losses[j] = float(score(selection_target, p)[0].mean())
            search.append(dict(step=len(targeted), candidate=j, loss=losses[j]))
        targeted.append(min(losses, key=losses.get))
    chosen['targeted'] = targeted
    return chosen, search


def controls():
    x = np.tile([[0.], [1.]], (12, 1))
    y = np.zeros((24, 32)); y[::2, [0, 16]] = 1; y[1::2, [1, 17]] = 1
    fit = ridge(x, y, .01)
    p, _ = forecast(x, fit)
    constant = np.full((24, 32), 1 / 16)
    q, _ = forecast(x, ridge(x, constant, .01))
    bad, invalid = repair(np.tile([-1., 2.] + [0.] * 14, (1, 2)))
    corrupt = False
    try:
        close([.2, .8], [.4, .6])
    except ValueError:
        corrupt = True
    teacher = np.zeros((24, 48)); teacher[:, [0, 32]] = 1
    teacher[::2, 16] = 1; teacher[1::2, 17] = 1
    selected, _ = selections(teacher, teacher, y, teacher, y, 5, size=1)
    tied, _ = selections(np.ones((24, 48)) / 16, np.ones((24, 48)) / 16,
        constant, np.ones((24, 48)) / 16, constant, 5, size=1)
    truth = np.zeros((1, 16)); truth[0, 0] = 1
    return {
        'live:known_separation': bool(np.max(abs(p - y)) < .002),
        'placebo:constant_target': bool(np.max(abs(q - constant)) < 1e-12),
        'live:informative_selection': selected['targeted'] == [1],
        'placebo:stable_selection_tie': tied['targeted'] == [0],
        'positive:repair_and_invalid_flag': bool(invalid.all() and np.all(bad > 0)
            and np.allclose(bad.reshape(1, 2, 16).sum(2), 1)),
        'positive:known_log_loss': abs(score(truth, np.ones((1, 16))/16)[0][0, 0] - np.log(16)) < 1e-14,
        'positive:corruption_rejected': corrupt,
    }


def array(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def features(history):
    seen, values = set(), []
    for o in history:
        c = o['context']
        values.extend(float(o['artifact'] == j) for j in range(16))
        values.extend(float(c[k] == v) for k, allowed in
            [('goal', (None, 0, 1)), ('signal', (None, 0, 1)), ('budget', (1, 2, 3))] for v in allowed)
        values.extend(float(c['offered'] is None or j in c['offered']) for j in range(4))
        values.extend([c['price_scale'], float(c['uninformative']), float(o['source'] not in seen)])
        seen.add(o['source'])
    return values


def references(root, cfg, pulse):
    records = json.loads(gzip.decompress((root/'evaluator/selection_points.json.gz').read_bytes()))
    indexed = {(r['split'], r['draw'], r['lineage'], r['index']): r for r in records}
    assert len(indexed) == len(records) == 4608
    output = {}
    for split, draw in product(('train', 'development'), cfg['training_draws']):
        a = array(root/'evaluator'/f'{split}-{draw}-artifact-history.npz')
        candidate = array(root/'evaluator'/f'{split}-{draw}-candidate.npz')
        close(a['ids'], candidate['ids'], 0)
        count = 16 if split == 'train' else 32
        lineages = cfg['train_lineages'] if split == 'train' else cfg['development_lineages']
        assert len(a['ids']) == count * len(lineages)
        assert {(int(l), int(i)) for l, i, _ in a['ids']} == set(product(lineages, range(count)))
        for j, (lineage, index, cl) in enumerate(a['ids']):
            if j % 128 == 0:
                pulse(phase='independent-portfolio-references', split=split, draw=draw, row=j)
            r = indexed[split, draw, int(lineage), int(index)]
            w = W.make_world(int(index) % 16, int(lineage))
            assert w == r['world']
            rule, _, endogenous, _ = W.FACTORS[int(index) % 16]
            assert cl == 2 * rule + endogenous
            matrix = np.concatenate([W.artifact_matrix(w, q) for q in cfg['candidate_queries']], 1)
            target = np.concatenate([W.artifact_matrix(w, q) for q in FAR_QUERIES], 1)
            likelihood = np.ones(24); seen = set()
            for o in r['history']:
                if o['source'] not in seen:
                    likelihood *= W.artifact_matrix(w, o['context'])[:, o['artifact']]
                    seen.add(o['source'])
            likelihood /= likelihood.sum()
            close(a['x'][j], features(r['history']), 0)
            close(candidate['teacher'][j], matrix[r['state']])
            close(candidate['exact'][j], likelihood @ matrix)
            close(a['old'][j], matrix[r['state'], :80])
            close(a['bank'][j], likelihood @ matrix[:, :80])
            close(a['target'][j], target[r['state']])
            close(a['exact'][j], likelihood @ target)
            close(a['prior'][j], target.mean(0))
        reader = array(root/'reader'/f'{split}-{draw}-artifact-history.npz')
        assert set(reader) == ({'x','old','target','ids'} if split == 'train' else {'x','ids'})
        for k, v in reader.items(): close(v, a[k], 0)
        if split == 'train':
            teacher_file = array(root/'training'/f'{draw}-candidates.npz')
            assert set(teacher_file) == {'x','target','teacher','ids'}
            for k in ('x','target','ids'):close(teacher_file[k], a[k], 0)
            close(teacher_file['teacher'], candidate['teacher'], 0)
        output[split, draw] = dict(a, teacher=candidate['teacher'], exact_bank=candidate['exact'])
    return output


def regroup(rows, cfg):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['portfolio'], r['bank'], r['world_class'], r['lineage']].append(r['loss'])
    assert all(len(v) == 8 for v in grouped.values())
    means = {k: float(np.mean(v)) for k, v in grouped.items()}
    draws = np.random.default_rng(190501).integers(8, size=(10000, 8))
    contrasts = []
    for bank, portfolio, cl in product(('learned','exact-oracle','noise-matched-oracle'),
                                      ('random','diversity','targeted'), (-1,0,1,2,3)):
        classes = range(4) if cl == -1 else [cl]
        delta = np.array([np.mean([means[portfolio,bank,k,l]-means['original',bank,k,l]
            for k in classes]) for l in cfg['development_lineages']])
        bounds = np.quantile(delta[draws].mean(1), [.025,.975])
        contrasts.append(dict(bank=bank,portfolio=portfolio,world_class=cl,mean=float(delta.mean()),
            low=float(bounds[0]),high=float(bounds[1]),lineages=8,lineage_values=delta.tolist()))
    fit_draw = []
    for draw, seed, portfolio, bank, cl in product(cfg['training_draws'],cfg['fit_seeds'],
        ('original','random','diversity','targeted'),('learned','exact-oracle','noise-matched-oracle'),(-1,0,1,2,3)):
        selected = [r for r in rows if r['draw']==draw and r['seed']==seed and r['portfolio']==portfolio
            and r['bank']==bank and (cl==-1 or r['world_class']==cl)]
        fit_draw.append(dict(draw=draw,seed=seed,portfolio=portfolio,bank=bank,world_class=cl,
            **{m:float(np.mean([r[m] for r in selected])) for m in ('loss','brier','invalid')}))
    return dict(contrasts=contrasts,fit_draw_strata=fit_draw,bootstrap_seed=190501,resamples=10000,
        population='eight paired development lineages; four equal classes, all draws/seeds/questions inside lineage',
        uncertainty='conditional on two training draws and two random-feature seeds; not confirmation')


def run(out, plan, pulse):
    checks = {k: bool(v) for k,v in controls().items()}
    write(out/'CONTROLS.json', checks)
    if not all(checks.values()):raise ValueError('independent controls failed before outcome access')
    for n,h in plan['design']['input_files'].items():assert file_digest(out/'inputs'/n)==h,n
    root = out/'inputs/original'; target=read(root/'PLAN.json');done=read(root/'COMPLETE.json');cfg=target['design']
    assert file_digest(root/'PLAN.json')==done['plan_sha256']==plan['design']['target_plan_sha256']
    assert file_digest(root/'COMPLETE.json')==plan['design']['target_complete_sha256']
    assert cfg['candidate_queries']==list(TRAIN_QUERIES+NEW_QUERIES) and cfg['ridge_grid']==[.001,.01,.1]
    assert cfg['train_lineages']==list(range(193000,193128)) and cfg['development_lineages']==list(range(190000,190008))
    assert cfg['training_draws']==[190201,190202] and cfg['fit_seeds']==[190101,190102]
    for n,h in {**done['files'],**done.get('execution_measurements',{})}.items():assert file_digest(root/n)==h,n
    data=references(root,cfg,pulse);rows=[];selection_checks=[];noise_checks=[];max_error=0.
    selections_saved={(r['draw'],r['seed']):r for r in read(root/'SELECTIONS.json')['selections']}
    inventory={(r['draw'],r['seed'],r['portfolio'],r['bank']):r for r in read(root/'SUMMARY.json')['fits']}
    for draw,seed in product(cfg['training_draws'],cfg['fit_seeds']):
        pulse(phase='independent-portfolio-fit',draw=draw,seed=seed)
        tr,te=data['train',draw],data['development',draw]
        fm=tr['ids'][:,0]<193064;sm=(tr['ids'][:,0]>=193064)&(tr['ids'][:,0]<193096);vm=tr['ids'][:,0]>=193096
        assert [int(x.sum()) for x in (fm,sm,vm)]==[1024,512,512]
        stem=f'{draw}-{seed}';acq=array(root/'models'/f'{stem}-acquisition.npz')
        rng=np.random.default_rng(seed);bw=rng.normal(size=(512,128))/np.sqrt(512);bb=rng.normal(scale=.2,size=128)
        close(bw,acq['basis_weights'],0);close(bb,acq['basis_bias'],0)
        h=np.tanh(np.pad(tr['x'],((0,0),(0,512-tr['x'].shape[1])))@bw+bb)
        ht=np.tanh(np.pad(te['x'],((0,0),(0,512-te['x'].shape[1])))@bw+bb)
        independent_head=ridge(h[fm],tr['teacher'][fm],.01)
        close(independent_head,acq['old_head'],1e-7)
        learned,_=forecast(h,acq['old_head']);learned_test,_=forecast(ht,acq['old_head'])
        chosen,search=selections(tr['teacher'][fm],learned[fm],tr['target'][fm],learned[sm],tr['target'][sm],seed)
        saved=selections_saved[draw,seed];assert chosen==saved['portfolios']
        assert [(r['step'],r['candidate']) for r in search]==[(r['step'],r['candidate']) for r in saved['search']]
        close([r['loss'] for r in search],[r['loss'] for r in saved['search']],1e-8)
        selection_checks.append(dict(draw=draw,seed=seed,portfolios=chosen,search=search))
        variance=np.mean((learned[fm]-tr['exact_bank'][fm])**2,axis=0);close(variance,acq['noise_variance'])
        rng=np.random.default_rng(seed+draw+900)
        noisy,_=repair(tr['exact_bank']+rng.normal(size=learned.shape)*np.sqrt(variance))
        noisy_test,_=repair(te['exact_bank']+rng.normal(size=learned_test.shape)*np.sqrt(variance))
        banks=array(root/'evaluator'/f'{stem}-banks.npz')
        for k,v in dict(learned_train=learned,learned_development=learned_test,noisy_train=noisy,noisy_development=noisy_test).items():close(banks[k],v)
        noise_checks.append(dict(draw=draw,seed=seed,requested_pre_repair_mse=float(variance.mean()),
            learned_fit_mse=float(np.mean((learned[fm]-tr['exact_bank'][fm])**2)),
            noisy_fit_post_repair_mse=float(np.mean((noisy[fm]-tr['exact_bank'][fm])**2))))
        pairs={'learned':(learned,learned_test),'exact-oracle':(tr['exact_bank'],te['exact_bank']),
            'noise-matched-oracle':(noisy,noisy_test)}
        for portfolio,indices in chosen.items():
            for bank,(x,xt) in pairs.items():
                pulse(phase='independent-portfolio-head',draw=draw,seed=seed,portfolio=portfolio,bank=bank)
                c=cols(indices);heads=[ridge(x[fm][:,c],tr['target'][fm],g) for g in cfg['ridge_grid']]
                values=[float(score(tr['target'][vm],forecast(x[vm][:,c],head)[0])[0].mean()) for head in heads]
                label=f'{stem}-{portfolio}-{bank}';model=array(root/'models'/f'{label}.npz')
                close(model['candidate_indices'],indices,0);close(model['ridge_grid'],cfg['ridge_grid'],0)
                close(model['validation_loss'],values,1e-8)
                choice=int(np.argmin(values));saved_choice=int(np.argmin(model['validation_loss']));assert choice==saved_choice
                close(model['target_head'],heads[choice],1e-7)
                p,bad=forecast(xt[:,c],model['target_head']);saved_forecast=array(root/'forecasts'/f'{label}.npz')
                max_error=max(max_error,close(p,saved_forecast['probabilities']))
                close(bad.astype(int),saved_forecast['invalid'].astype(int),0);close(te['ids'],saved_forecast['ids'],0)
                inv=inventory[draw,seed,portfolio,bank]
                assert inv['ridge']==cfg['ridge_grid'][choice] and inv['target_parameters']==81*32 and inv['old_parameters']==129*128
                assert [inv[k] for k in ('fit_histories','portfolio_histories','ridge_histories','acquired_queries','deployed_queries')]==[1024,512,512,8,5]
                loss,brier=score(te['target'],p)
                for lineage,cl,q in product(cfg['development_lineages'],range(4),range(2)):
                    mask=(te['ids'][:,0]==lineage)&(te['ids'][:,2]==cl);assert int(mask.sum())==8
                    rows.append(dict(draw=draw,seed=seed,portfolio=portfolio,bank=bank,lineage=lineage,world_class=cl,
                        query=q,histories=8,loss=float(loss[mask,q].mean()),brier=float(brier[mask,q].mean()),invalid=float(bad[mask,q].mean())))
    raw=json.loads(gzip.decompress((root/'raw/portfolio_points.json.gz').read_bytes()))
    keys=('draw','seed','portfolio','bank','lineage','world_class','query');lookup={tuple(r[k] for k in keys):r for r in raw}
    assert len(lookup)==len(rows)==3072
    for r in rows:
        original=lookup[tuple(r[k] for k in keys)];assert original['histories']==8
        close([r[k] for k in ('loss','brier','invalid')],[original[k] for k in ('loss','brier','invalid')])
    result=regroup(rows,cfg);summ=read(root/'SUMMARY.json')
    for a,b in zip(result['contrasts'],summ['contrasts']):
        assert all(a[k]==b[k] for k in ('portfolio','bank','world_class','lineages'))
        close([a[k] for k in ('mean','low','high')],[b[k] for k in ('mean','low','high')])
    spectral=read(root/'SPECTRA.json')['rows'];assert len(spectral)==2048
    for row in spectral:
        indices=selections_saved[row['draw'],row['seed']]['portfolios'][row['portfolio']]
        w=W.make_world(row['cell'],row['lineage'])
        matrix=np.concatenate([np.ones((24,1))]+[W.artifact_matrix(w,cfg['candidate_queries'][j]) for j in indices],1)
        s=np.linalg.svd(matrix,compute_uv=False);close(s,row['singular_values'])
        assert sum(s>1e-10)==row['rank_1e10'] and sum(s>1e-12)==row['rank_1e12']
    write(out/'INDEPENDENT_REGROUP.json',result)
    write(out/'SELECTION_RECONSTRUCTION.json',dict(selections=selection_checks,noise_checks=noise_checks))
    (out/'review_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    checks['positive:complete_reconstruction']=True
    return dict(controls=checks,rows=len(rows),reference_histories=4608,target_heads=48,acquisition_heads=4,
        spectra=len(spectral),contrasts=len(result['contrasts']),max_saved_forecast_error=max_error,
        target_plan_sha256=plan['design']['target_plan_sha256'],
        scope='independent old-world method verification; no new fit setting/population; scientific classification remains event-review duty')
