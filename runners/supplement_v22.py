"""Descriptive finite V22 contrasts from retained forecasts and world means."""
import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report-module', type=Path, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--aggregates', type=Path, required=True)
    p.add_argument('--cost', type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('report', args.report_module)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    start = time.process_time()
    try:
        # All declared weights are crossed; zero-weight identical controls stay.
        a = report.read(args.aggregates / 'a3/ANALYSIS.json')['estimates']
        pairs = []
        for coupled, prefix, benign in itertools.product((False, True), (4, 8, 16), (False, True)):
            label = f'coupled{coupled}/prefix{prefix}/benign{benign}'
            for policy, wd, wg in itertools.product(('withdraw-.7', 'withdraw-.9'), (0., .25, .5, .75, 1.), (0., .25, .5, .75, 1.)):
                row = dict(coupled=coupled, prefix=prefix, benign=benign, withdrawal=policy,
                           withdrawal_weight=wd, guard_weight=wg, metrics={})
                for metric in ('purpose', 'value', 'technique', 'recognition', 'flag'):
                    base, withdraw, guard = [np.array(a[label + '/' + pol + '/' + metric]['world_values'])
                                             for pol in ('no-gate', policy, 'early-guard')]
                    row['metrics'][metric] = report.interval(wd*(withdraw-base)-wg*(guard-base))
                row['technique_matched'] = abs(row['metrics']['technique']['mean']) <= .02
                row['identical_no_gate_control'] = wd == wg == 0
                pairs.append(row)
        report.write(args.aggregates / 'a3/MIXTURES.json', dict(
            definition='Each policy is mixed with no gate at fixed 0/.25/.5/.75/1 weights; all withdrawal/guard weight pairs retained. Differences are withdrawal mixture minus guard mixture.',
            match_rule='absolute world-mean technique-production accuracy difference <= .02; this descriptive matching is not a confidence-interval equivalence test',
            all_pairs=len(pairs), matched=[r for r in pairs if r['technique_matched']],
            unmatched=len([r for r in pairs if not r['technique_matched']]),
            all_pairs_sha256=__import__('hashlib').sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()))
        for card in ('A1', 'B1', 'C2', 'A2', 'A3', 'B2', 'C1'):
            packet = next(x for x in (args.campaign / 'packets').iterdir() if report.read(x/'PLAN.json')['card']==card)
            s = report.Study(packet)
            if card == 'A1':
                for k, goal, competence in itertools.product((0, 2, 8, 32), (0, 1), (.35, 1.2)):
                    sel=dict(condition='native', k=k, source_goal=goal, competence=competence)
                    for j in range(13):
                        s.measure(f'curve/k{k}/goal{goal}/source-noise{competence}/D{j*.5:g}', lambda w,sel=sel,j=j: np.mean([w['details'][i]['shift_curve'][j] for i in s.ids(w,sel)]))
                for sr in (.4, 1.2):
                    for j in range(13):
                        s.measure(f'expertise/noise{sr}/D{j*.5:g}', lambda w,sr=sr,j=j: np.mean([w['details'][i]['shift_curve'][j] for i in s.ids(w,dict(condition='native',expertise=sr))]))
                for k in (0,2,8,32):
                    for label in (.2,.54,.9):
                        s.measure(f'steering-detection/k{k}/label{label}', lambda w,k=k,label=label: np.mean([d['steering_probability']>=.9 for d,r in zip(w['details'],w['records']) if d['condition']=='native' and d['k']==k and d['source_goal']==1 and r['public']['label']==label]))
                for condition in ('native','unrelated','duplicated'):
                    for m in ('signed_error','belief_shift'):
                        s.measure(f'{condition}/{m}',lambda w,c=condition,m=m:np.mean([w['details'][i][m] for i in s.ids(w,dict(condition=c))]))
            if card == 'B1':
                # Matched shift changes by reader/dimension/history/control.
                for d,k,c in itertools.product((2,3,4),(2,8,32),('native','zero-cost','independent-effort','alias')):
                    for reader in s.worlds[0]['scores']:
                        if reader.endswith(('-0','-1')): continue
                        for m in ('log_loss','brier','accuracy','confident_error'):
                            s.measure(f'd{d}/k{k}/{c}/{reader}/{m}',lambda w,d=d,k=k,c=c,r=reader,m=m:s.mean(w,r,dict(d=d,k=k,condition=c,shift=True),0,m)-s.mean(w,r,dict(d=d,k=k,condition=c,shift=False),0,m))
            if card == 'C2':
                # Similarity factors crossed separately, not silently pooled.
                for k,sp in itertools.product((0,2,8,32),(False,True)):
                    sel=dict(k=k,coupled=False,disclosed=False,unrelated=False,share_pref=sp)
                    for head,target in enumerate(('operation','preference')):
                        s.measure(f'k{k}/share-pref{sp}/{target}/library-benefit',lambda w,sel=sel,h=head:s.mean(w,'law-marginal',dict(sel,share_lib=False),h)-s.mean(w,'law-marginal',dict(sel,share_lib=True),h))
                for k,sl in itertools.product((0,2,8,32),(False,True)):
                    s.compare(f'k{k}/share-lib{sl}/records-benefit',dict(k=k,coupled=False,disclosed=False,unrelated=False,share_lib=sl),1,'without-records')
            if card == 'A2':
                for signature,swapped in itertools.product((False,True),(False,True)):
                    sel=dict(signature=signature,swapped=swapped,source_goal=1)
                    for m in ('technique_production','goal_probability'):
                        s.measure(f'signature{signature}/swapped{swapped}/exact-minus-missing/{m}',lambda w,sel=sel,m=m:np.mean([w['details'][i][m]['law-marginal']-w['details'][i][m]['missing-process'] for i in s.ids(w,sel)]))
            if card == 'C1':
                for b,cov in itertools.product((2,3),(.25,.5,.75)):
                    def delta(w,b=b,cov=cov):
                        vals=[]
                        for i in s.ids(w,dict(branching=b,coverage=cov)):
                            d=w['details'][i]; y=w['y'][i,d['focus']]
                            vals.append(-np.log(d['ablations']['adjacent'][y])+np.log(d['ablations']['distant'][y]))
                        return np.mean(vals)
                    s.measure(f'branch{b}/coverage{cov}/adjacent-minus-distant-removal-loss',delta)
            # Every primary stratum gets paired direct-minus-exact intervals and
            # same-cell metric reversals. Existing immutable HEADROOM stays authoritative.
            c=report.read(packet/'original/CONTRASTS.json')
            headroom=[r for r in c['matched_headroom'] if r['budget']==8192]
            reversals=[]
            for row in headroom:
                sel=row['group']; target=row['target']
                if card=='C1':
                    b=sel['branching']; depth=int(target[-1]); heads=list(range(sum(b**d for d in range(depth)),sum(b**d for d in range(depth+1))))
                else:
                    heads=[{'A1':['property'],'B1':['future-choice'],'C2':['operation','preference'],'A2':['content-property','top-goal','technique'],'A3':['purpose','value','technique'],'B2':['acquisition-route','future-choice']}[card].index(target)]
                metrics={m:report.interval([s.mean(w,'direct-8192',sel,heads,m)-s.mean(w,'law-marginal',sel,heads,m) for w in s.worlds]) for m in ('log_loss','brier','accuracy','confident_error')}
                reversals.append(dict(group=sel,target=target,disposition=row['disposition'],metrics=metrics))
            report.write(args.aggregates/card.lower()/'SUPPLEMENT.json',dict(estimates=s.out,paired_direct_8192=reversals,
                         source_sha256=report.digest(__file__),scope='retained-data descriptive contrasts; no scientific rerun or new criterion'))
            print(card,'supplement',len(s.out),len(reversals),flush=True)
    finally:
        report.write(args.cost,dict(cpu_seconds=time.process_time()-start,source_sha256=report.digest(__file__),kind='retained-data supplementary contrasts'))


if __name__=='__main__':
    main()
