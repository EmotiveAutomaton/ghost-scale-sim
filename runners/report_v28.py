"""File V28's three finite reports from verified aggregates and retained checks."""
import argparse,json,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from ghostscale.validation.soundingline.v28.common import read,write,digest,median

TYPES=('ONE','POINT','BAND','MOVE','DULL','SHARP')
QUESTIONS={
 'G1':'What must a reader know about a maker\'s habits, and when does another probabilistic pass help?',
 'G2':'Which set-point rules can a visible record distinguish, and how much record is needed?',
 'G3':'When can choices distinguish a resource held as a target from a tool or a reserve?'}
G2_FINDING=("Which set-point rules can a visible record distinguish, and how much record is needed? "
 "Across the 28-cell planning map, the one-sided defender remains unresolved against the dulling type "
 "in 63.75% of ordinary 256-step paths. A prolonged deficit separates them at a median statistic of "
 "86 (95% interval 84–91), with 0.25% unresolved. Extending the ordinary record to 1,024 steps also "
 "separates them: median 817.5 (694–917), with 34% unresolved. The apparent alias is therefore "
 "horizon-dependent, not an exact equivalence of these positive-rate models. This is a descriptive "
 "constructed-method scale map, not evidence about people.")
G3_FINDING=("When can choices distinguish a resource held as a target from a reserve? "
 "In the 40-cell map, target-holder 3 separates from reserve 20 in the plentiful setting "
 "at a median statistic of 16 (95% interval 14–19), with no unresolved worlds. "
 "In the scarce setting, 65% remain unresolved after 512 steps. A surplus choice with an "
 "unmet need occurs in 100% versus 37.75% of worlds. These are descriptive constructed-method "
 "scales with initial resource uniform over 0–30, not human evidence.")

def fmt(x):
    if isinstance(x,str):return 'censored' if x=='infinite' else x
    return f'{x:.6g}'

def interval(r):return ' to '.join(fmt(x) for x in r['interval95'])

def text(path,value):path.write_text(value,encoding='utf-8',newline='\n')

def truth(s):
    t,theta,pi,extra=s['truth'];name=TYPES[int(t)]
    return name+(f'({extra:g})' if t>=2 else '')

def setting_label(family,s):
    if family=='G1':return f"{s['size']}, {s['overlap']}, goal share {s['rho']:g}, history {s['K']}"
    if family=='G2':return f"{truth(s)}, {s['regime']}, {s['steps']} steps"
    return f"inflow {s['q_in']:g}, need {s['q_need']:g}, truth {('TOOL2','TOOL4','SAVER10','SAVER15','SAVER20','CAPTURED3','CAPTURED5')[s['truth']]}"

def all_rows(family,rows):
    labels={'last_below_plus_one':'sustained-separation statistic',
      'not_separated_share':'unresolved world share','fitted_theta':'mean fitted reference',
      'fitted_pi':'mean fitted precision','loss_per_step':'mean in-sample loss, nats per step',
      'fit_pair_share':'share with fitted reference/precision',
      'gap_to_J':'joint-reader minus reader log score, nats',
      'log_score':'mean true-goal log score, nats',
      'true_goal_below_one_percent':'true goal below 1% share','accuracy':'goal accuracy',
      'top_probability_minus_accuracy':'top probability minus accuracy',
      'departure_auc':'token-source discrimination AUC',
      'PeL4_minus_Pe_log_score':'four-loop minus one-pass log score, nats',
      'telling_moments_before_separation':'telling moments before separation',
      'worlds_with_telling_moment':'worlds with a telling moment',
      'share_of_steps_with_telling_moment':'mean share of telling steps',
      'weight_3_1_against_1_3':'weight (3,1) against (1,3) separation statistic'}
    out=['\nComplete map. Each row is one setting and measure; intervals are pointwise 95%, not multiplicity-adjusted. Censored values are retained beyond the observation horizon.\n',
      '| Setting | Reader/rival or reserve | Measure | Estimate | 95% interval | Scale only |',
      '|---|---|---|---|---|---|']
    for r in rows:
        who=r.get('reader',r.get('rival',str(r.get('reserve',''))))
        if r['measure']=='fit_pair_share':who=f"reference {r['fitted_theta']:g}; precision {r['fitted_pi']:g}"
        out.append(f"| {setting_label(family,r['setting'])} | {who} | {labels[r['measure']]} | {fmt(r['estimate'])} | {interval(r)} | {str(r['scale_only']).lower()} |")
    return '\n'.join(out)+'\n'

def g1_tables(rows):
    out=['\nFirst table: one-page loop cost, the first sampled history meeting each descriptive comparison, typical-habit loss after 16 earlier documents, and paired token-mark discrimination. Threshold crossings below use point estimates, not a claim test.\n',
      '| World; overlap; goal share | Four-loop minus one-pass at no history (95% interval), nats | First history: loop beats one pass | First history: one pass within 0.05 of joint | Library-pass gap at no history; history 16 | Typical-habit gap at history 16 | AUC, history 0 → 16: own / joint / typical |',
      '|---|---|---|---|---|---|---|']
    groups=sorted({(r['setting']['size'],r['setting']['overlap'],r['setting']['rho']) for r in rows})
    for size,overlap,rho in groups:
        rr=[r for r in rows if (r['setting']['size'],r['setting']['overlap'],r['setting']['rho'])==(size,overlap,rho)]
        def find(reader,measure,k):return next(r for r in rr if r.get('reader')==reader and r['measure']==measure and r['setting']['K']==k)
        ks=sorted({r['setting']['K'] for r in rr});c=find('contrast','PeL4_minus_Pe_log_score',0)
        first=lambda pred:next((str(k) for k in ks if pred(k)),'beyond tested history')
        beat=first(lambda k:find('contrast','PeL4_minus_Pe_log_score',k)['estimate']>0)
        near=first(lambda k:find('Pe','gap_to_J',k)['estimate']<=.05)
        ps='; '.join(fmt(find('Ps','gap_to_J',k)['estimate']) for k in (0,16))
        auc=' / '.join(' → '.join(fmt(find(r,'departure_auc',k)['estimate']) for k in (0,16)) for r in ('Pe','J','Pp'))
        out.append(f"| {size}; {overlap}; {rho:g} | {fmt(c['estimate'])} ({interval(c)}) | {beat} | {near} | {ps} | {fmt(find('Pp','gap_to_J',16)['estimate'])} | {auc} |")
    return '\n'.join(out)+'\n'


def g1_finding(rows):
    losses=[-r['estimate'] for r in rows if r['measure']=='PeL4_minus_Pe_log_score' and r['setting']['size']=='small' and r['setting']['K']==0]
    assert len(losses)==6 and min(losses)>0
    near={size:sum(r['estimate']<=.05 for r in rows if r.get('reader')=='Pe' and r['measure']=='gap_to_J' and r['setting']['size']==size and r['setting']['K']==16) for size in ('small','large')}
    return (QUESTIONS['G1']+f" In the 60-cell map, four passes using only one page hurt all six small-world settings, losing {min(losses):.3f} to {max(losses):.3f} nats of true-goal log score versus one pass. With 16 earlier documents, the own-history one-pass reader comes within 0.05 nats of the joint reader in {near['small']} of six small-world settings and {near['large']} of six larger-world settings. These are descriptive constructed-method results, not evidence about people or automatic engine admission.")

def g2_tables(rows,original):
    out=['\nEvents that separate each pair at 256 steps. N is ordinary drift; D blocks restoration on steps 64–127; W adds a windfall then. A finite median counts as resolved; this is descriptive, with uncertainty in the complete table. Both-D-and-W is retained as its own category.\n',
      '| Truth (rate or band width in parentheses) | Rival | N median | D median | W median | Telling regime |',
      '|---|---|---|---|---|---|']
    selected=[r for r in rows if r['measure']=='last_below_plus_one' and r['setting']['steps']==256]
    for key in sorted({(truth(r['setting']),r['rival']) for r in selected}):
        rr={r['setting']['regime']:r for r in selected if (truth(r['setting']),r['rival'])==key}
        finite={k for k,r in rr.items() if r['estimate']!='infinite'}
        label='resolved under N' if 'N' in finite else ('under D and W' if finite=={'D','W'} else ('only under '+next(iter(finite)) if finite else 'under nothing'))
        out.append('| '+' | '.join([*key,*[fmt(rr[k]['estimate']) for k in ('N','D','W')],label])+' |')
    out+=['\nSimple POINT-reader symptoms under ordinary drift. Each cell has true reference 12 and initial precision 1. The fitted medians and their order-statistic intervals are reconstructed from retained measures; complete mean and fit-frequency rows follow.\n',
      '| Truth | Steps | Fitted reference median (95%) | Fitted precision median (95%) | Mean in-sample loss, nats/step (95%) |',
      '|---|---|---|---|---|']
    from ghostscale.validation.soundingline.v28.g2 import settings
    for index,s in enumerate(settings()):
        if s['regime']!='N':continue
        with np.load(original/f'cell-{index:03d}_measures.npz',allow_pickle=False) as z:
            theta=median(z['fitted_theta']);pi=median(z['fitted_pi'])
        loss=next(r for r in rows if r['setting']==s and r['measure']=='loss_per_step')
        out.append(f"| {truth(s)} | {s['steps']} | {fmt(theta['estimate'])} ({interval(theta)}) | {fmt(pi['estimate'])} ({interval(pi)}) | {fmt(loss['estimate'])} ({interval(loss)}) |")
    return '\n'.join(out)+'\n'

def held_report(family,c):
    gate=c/'families'/family/'PREFLIGHT.json'
    if gate.exists() and read(gate)['passed']:
        return (f'# {family}: admitted execution\n\n'+QUESTIONS[family]+'\n\n'
          'The owner-approved pre-flight admission is recorded. Known-answer checks passed, '
          'and the original, replay and independent reconstruction are built under unchanged '
          'scientific sources. Full verification is still in progress; no scientific finding '
          'is reported from the partial map. The original failed comparison remains retained.\n')
    comparison=read(c/'families'/family/'PREFLIGHT_CANDIDATE.json')
    failed=[x for x in comparison['checks'] if not x['passed']]
    out=[f'# {family}: admission held\n',QUESTIONS[family]+'\n',
      f"All known-answer checks passed. The excluded pre-flight comparison recorded {len(failed)} outside-tolerance entries out of {len(comparison['checks'])}. No original scientific map, seed replay or whole-grid reconstruction has been admitted. This is not a mechanism null.\n"]
    if family=='G1':
        out+=['The original comparison is retained, including five AUC entries diagnosed as inapplicable summary ranges. The supplied 0.58–0.68 range comes from the discarded 0.1 goal-share cells; the actual first output gives low-overlap, goal-share 0.2 AUC 0.845 and goal-share 0.4 AUC 0.894. Our excluded values are 0.840756 and 0.908935. The older high-overlap generator also allowed goals to overlap one another; the current packet requires disjoint target sets. These are baseline-scope differences, not evidence for altering the current mark formula.\n',
          'The remaining 18 entries are score gaps. A typical-habit reader ignores history, while the supplied comparison resamples libraries at each history length. For example, its large high-overlap, goal-share 0.1 values at histories 1/4/16 are 5.36/4.99/6.12 nats. Our paired excluded estimate is 5.54846 at each, with library-block standard error 0.289829. The owner decision on an uncertainty-aware comparison remains pending; fixed tolerances have not been waived.\n']
    else:
        out+=['The failed cell has inflow 0.5, need probability 0.3 and a target-holder truth against reserve 10, using the supplied fixed initial resource of 5. Our 400 excluded worlds have 44.25% unresolved versus the supplied 200-world estimate of 39%; the permitted difference is five percentage points. A retained diagnostic reproduces the supplied 39% exactly, with identical censored-path decisions and maximum final log-odds disagreement 1.71e-13. The independent kernel also agrees. The fresh-sample difference is 1.24 combined standard errors. The original comparison remains failed pending the owner decision on exact-path admission.\n']
    out+=['Outside-tolerance entries below are excluded validation data, not scientific findings.\n',
      '| Setting / comparison | Measure or reader | Own estimate | Supplied value/range | Original tolerance |',
      '|---|---|---|---|---|']
    for x in failed:
        setting=setting_label(family,x['setting']) if 'setting' in x else x['name']
        measure=x.get('reader','')+' '+x.get('measure','')
        out.append(f"| {setting} | {measure.strip()} | {fmt(x['actual'])} | {x['expected']} | {x['tolerance']} |")
    out+=['\nEvery excluded raw path and the failed receipt are retained. No seed search, extra science or threshold tuning is authorized by unused capacity.\n']
    return '\n'.join(out)

def file_reports(c,repo,g1_repair=None):
    started=time.process_time();docs=repo/'docs/versions/v28-maps';results=repo/'results/v28'
    findings={};states={};base=c
    for family in ('G1','G2','G3'):
        c=g1_repair if family=='G1' and g1_repair is not None else base
        source=read(c/'SOURCE.json')
        dest=results/family;dest.mkdir(exist_ok=True)
        shutil.copyfile(c/'jobs'/(family+'-controls')/'output/CONTROLS.json',dest/'CONTROLS.json')
        for name in ('PREFLIGHT_CANDIDATE.json','PREFLIGHT_DIAGNOSIS.json','PREFLIGHT_UNCERTAINTY.json','PREFLIGHT.json','VERIFIED.json'):
            src=c/'families'/family/name
            if src.exists():shutil.copyfile(src,dest/name)
        if not (c/'families'/family/'VERIFIED.json').exists():
            states[family]='ADMITTED' if (c/'families'/family/'PREFLIGHT.json').exists() and read(c/'families'/family/'PREFLIGHT.json')['passed'] else 'HELD_PREFLIGHT'
            text(docs/'reports'/(family+'.md'),held_report(family,c));continue
        verification=read(c/'families'/family/'VERIFIED.json');original=c/'jobs'/(family+'-original')/'output'
        if digest(original/'SUMMARY.json')!=verification['original_summary_sha256']:raise ValueError('verified summary changed')
        for name in ('SUMMARY.json','RAW_MANIFEST.json'):shutil.copyfile(original/name,dest/name)
        rows=read(original/'SUMMARY.json')['rows'];states[family]='VERIFIED'
        if family=='G2':
            finding=G2_FINDING
            body=(f'# G2: visible-stock planning map\n\n{finding}\n\n'
              'Implementation choices: the single 1,024-step ordinary cell is ONE(12,1), frozen before outcomes. Moving references use the previous pre-action stock; counters include the current stock. Stock updates follow action, disturbance, windfall and clipping.\n\n'
              'There are 28 settings, 400 independent worlds each, 27 at 256 steps and one at 1,024. Every row is scale only. Exact type evidence integrates each parameter grid with equal type priors. A separately written scalar kernel reconstructs the complete grid.\n\n'
              'A spend rules out no-spend types. Identical conditional action laws give zero conditional evidence increments; differently learned parameter distributions can still give different marginal increments. Exogenous disturbances carry no type evidence. Positive 0.02 rates create near-aliases over short records, not exact infinite-history aliases.\n\n'
              'The separation statistic is literally one plus the last zero-based step below one nat. It is not the first crossing. Zero means above the bar after the first observation; a final below-bar path is right-censored. Median intervals preserve censoring. Fitted-model losses are in-sample comparisons, so a fitted correct-class model can slightly outperform its true parameter on that same sample. Means use world-level t intervals, proportions Wilson intervals, and medians binomial order-statistic intervals. All are pointwise descriptive.\n')
            body+=g2_tables(rows,original)
        elif family=='G1':
            finding=g1_finding(rows)
            body=f'# G1: maker history and joint reading\n\n{finding}\n\nHistory lengths share the same page, maker, library and nested documents. Every standard error uses whole library blocks; pooled token AUC uses midranks and delete-one-block jackknife uncertainty. Exact joint inference has the expected proper-score ceiling, full-library beams equal it, and the typical-habit reader is invariant to history. Those identities do not fix the measured absolute performance levels.\n'
            body+=g1_tables(rows)
        else:
            finding=G3_FINDING
            body=f'# G3: resources as tools, reserves or targets\n\n{finding}\n\nPlentiful means inflow probability 0.8 and need probability 0.15; scarce means 0.3 and 0.3. Initial resource is uniform over integers 0–30 in science; the supplied fixed-5 condition is excluded pre-flight only. Inflow is capped before choice. No-need steps have likelihood one for every class. Conditional on a common weight pair, target-holder 3 and a saver agree below that saver\'s reserve; marginal class odds can move later because their weight posteriors differ. This is a planning scale map, not a human classification result. Telling-moment counts retain censored paths and count observations strictly before sustained separation.\n'
        findings[family]=finding
        if family=='G1':body+='\nAdmission retains the original fixed-tolerance failure. The owner approved max(original tolerance, three combined library-block standard errors), using the supplied scripts and seeds to reconstruct baseline uncertainty. All 176 score means reproduce at printed precision and all 292 applicable comparisons pass. Sixteen old-setting AUC ranges are inapplicable; eight comparable low-overlap points use the supplied original output. That admission left the exact known-answer gates and scientific source unchanged; the subsequent numerical repair is described below.\n'
        if family=='G3':body+='\nAdmission retains the failed fresh-sample check: 44.25% unresolved versus the supplied 39%, beyond the original five-point tolerance. The owner approved the exact retained-path comparison instead: every censored-path decision agrees, with maximum final log-odds error 1.71e-13. This admission amendment does not change the scientific initial-condition law or any known-answer gate.\n'
        if family=='G1' and g1_repair is not None:
            resolution=max(r.get('resolution_10_vs_12_absolute_difference',0) for r in rows)
            body+=f'\nNumerical repair: the original whole-grid reconstruction failed on empirical goal ties and token ranks, and is preserved. Exact integer-count symmetries now receive equal goal probabilities, with the lowest goal index breaking the tie. Mark scores use twelve decimal places before exact midranking; the largest change in AUC when using ten places is {resolution:.3g}. The same seeds and grid were rerun in a separate capsule, with unchanged verification tolerances and a repeated 292-comparison pre-flight check. This repairs numerical representation; it does not add worlds or change reader laws. The architectures remain untested miniatures.\n'
            shutil.copyfile(c/'REPAIR_PROTOCOL.json',dest/'REPAIR_PROTOCOL.json')
            shutil.copyfile(Path(source['root'])/'SOURCE_MANIFEST.json',dest/'SOURCE_MANIFEST.json')
            write(dest/'SOURCE_BINDINGS.json',{k:v for k,v in source.items() if k!='root'},immutable=False)
        body+=all_rows(family,rows)
        gates=read(dest/'CONTROLS.json')['gates']
        body+=f'\nValidation: {len(gates)} known-answer checks pass; the admitted pre-flight receipt, full byte replay and independent whole-grid reconstruction are retained in [results](../../../../results/v28/{family}/). Source manifest SHA-256: `{source["manifest_sha256"]}`. No raw world data is published.\n'
        if family=='G2':body+='\nSounding Line route: set-point rules 2, 5 and 7; this is a list of record lengths, intervention conditions and simple-reader symptoms to consider before interpreting human fits. ToMpathy: no direct engine change is admitted by this map.\n'
        if family=='G1':body+='\nSounding Line route: whose personal history predicts departures from habit. ToMpathy route: compare typical-habit, own-history and probabilistic joint reading under Round 12 test 9; this does not authorize an engine change.\n'
        text(docs/'reports'/(family+'.md'),body)
    c=base;source=read(c/'SOURCE.json')
    shutil.copyfile(Path(source['root'])/'SOURCE_MANIFEST.json',results/'SOURCE_MANIFEST.json')
    write(results/'SOURCE_BINDINGS.json',{k:v for k,v in source.items() if k!='root'},immutable=False)
    account=c/'accounting/reporting.json';previous=read(account).get('cpu_seconds',0.) if account.exists() else 0.
    write(account,dict(family='G1',state='STOPPED',cpu_seconds=previous+time.process_time()-started),immutable=False)
    return states,findings

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--g1-repair',type=Path)
    a=p.parse_args();states,findings=file_reports(a.campaign.resolve(),a.repo.resolve(),a.g1_repair.resolve() if a.g1_repair else None)
    print(json.dumps(dict(states=states,findings=findings),indent=2))
