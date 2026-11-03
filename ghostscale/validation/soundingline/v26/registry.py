"""Prospective card grids, finite variants and declared analysis choices."""
from itertools import product
from .contract import ORDER,FAMILY_HOURS

NAMES=dict(H1='HIERARCHY_TOP',H2='PAUSES_DIFFICULTY_DELIBERATION',H3='GLOBAL_REVISION',E1='TERMINAL_OR_CIRCULATING',E2='ONE_MODEL_TWO_DIRECTIONS',E3='FEELINGS_AS_CACHED_SOLUTIONS',U1='OPACITY_GATE',R1='W1_REPAIR',R4='W4_PRECISION',R13='W13_REPAIR')
MODULES=dict(H1='hierarchy',H2='pauses',H3='revision',E1='values',E2='self_model',E3='cached',U1='opacity',R1='alias_repair',R4='precision_repair',R13='persistence_repair')
ROUTES={
 'H1':('Sounding Line T3/T5: own-library mismatch and records needed at the hierarchy top.','ToMpathy expertise/habit top tier: constructed benchmark only; real-record marks remain untested.'),
 'H2':('Sounding Line S5/S6: goal evidence in pauses after modelling choice difficulty.','ToMpathy current-attention marks require an explicit difficulty model.'),
 'H3':('Sounding Line S2/S3: additional goal evidence in revision history per event.','ToMpathy expertise/habit panel: constructed support for realigned spans, not a real-record validation.'),
 'E1':('Sounding Line V3/V4: forced-use analogues and longitudinal records needed.','ToMpathy core values: distinguish fixed weights from drifting usage ratios under explicit assumptions.'),
 'E2':('Sounding Line T5: when own-model versus population anchoring helps.','ToMpathy prior design: own model, population or mixture, subject to real-data validation.'),
 'E3':('Sounding Line V3: drive cues may identify cached cue maps rather than ends.','ToMpathy core values drive channels must state which generator reading is supported.'),
 'U1':('Sounding Line U1: source-model observables for co-writing tests.','ToMpathy: no route this week.'),
 'R1':('Sounding Line: no route this week.','ToMpathy owner-question order: corrected exact-alias intervention map.'),
 'R4':('Sounding Line: no route this week.','ToMpathy B15: replace or confirm the records-needed curve after precision repair.'),
 'R13':('Sounding Line: shifted-episode counts needed to distinguish persistence from proxy learning.','ToMpathy: no route this week.')}
CHOICES={
 'H1':'Six makers, one per top goal per world. Dirichlet piece mixtures with concentration2; common-overlap concentration8, 90% shared and10% extra support. Seeded strings2-4 steps, no exclusive-step acceptance. Fixed10,000 training documents per top goal per world, separate training seed, before observations; add-one bigrams. Hidden piece-count is uniform1-3. Four/eight/sixteen document prefixes share trajectories. Piece-reader retains the total-count distribution but removes shared-subgoal dependence. Expert libraries replace a fixed last quarter/half of piece strings. Novel strings have the stipulated .05 unknown channel.',
 'H2':'Six makers per world, each goal once with cyclic habits. Goal range .5-2, after the full-range check; narrower engineering check retained separately. Observed replacements identify a redraw even if the replacement equals the old unit. Primary is total gain divided by total deliberations within a world, not mean case ratios. Misread reader uses q=0.',
 'H3':'Six makers, one per initial goal, cyclic habits. Documents share initial goal and habit. Settled goal is .5 same or .1 each other; discovery equally half/end. r prior uniform .3/.6/.9, shared across all documents of one maker. Prefixes1/2/4/8/16. Revise flag includes redraws with unchanged content. Primary pooled gain/event within a world. Local settled goals are marginalized before accumulating maker evidence.',
 'E1':'One maker of each type per world, uniformly sampled from eligible start grid; equal type priors and uniform within-type finite states. B2 begins at prior counts2 in every dimension. Four environment mixes separately compare stationary, forced every8, eight-record devaluation from64, shift from128, and a separate trade ladder. Forced choices update state with unit likelihood. Exact source histories, not future outcomes, update the predictor. Usage predictor uses last16 outcomes and uniform fallback on zero total.',
 'E2':'Four pairs per distance bin per world, uniform maker grid and uniform reader within bin. Known beta3, no habit in this W1-style choice variant. Partial mismatch replaces effect rows of final pieces with independent grid outcomes. S/P contrast uses posterior log scores, not gains relative to different priors. Mixture uses half of each prior with the own library.',
 'E3':'Eight makers per world. Independent fair binary cues; block probability .02+.16*feature0, mean .1. Signature precedes cache update; cache updates after every observed record. Cache-map inference marginalizes the finite learning rates using generator evidence; it is not an independently identifiable hidden map when rate and full history are supplied. End posterior assumes beta3 and no habit. Primary is mean total end-score difference per whole world; per-outcome-block rates also retained.',
 'U1':'Thirty-two makers, balanced true hypotheses; source strength fixed across track and claims. Track records reveal all eight pool items. Posterior on source strength integrated jointly across all new claims, never resampled per claim. Menu arm reveals four random pool items plus selected sign, using exact hypergeometric marginal. Records-needed uses h0-world mean capture curve, not individual hits. Design-check share averages the two biased source strengths equally.',
 'R1':'Eight sampled split-alias pairs per world, alternating their true endpoints. Full W1 210-hypothesis reader retained. Five training pieces share values on ends0/1; sixth differs. Separate adaptive remove/satiate/switch tracks up to16 interventions, each compared with equal-count passive records. Ties in expected information choose the first declared probe.',
 'R4':'Retained W4 alpha10, non-placebo setting0, sixteen original world identities. Original8/32/128/512/1024 increments regenerated unchanged. Extra levels split increments with a separately seeded multivariate hypergeometric bridge. Individual reader only; arm a original half-even rule with boundary fallback; b/c rank -log complementary tail checked at50/80 decimal digits. These are repairs on retained worlds, not fresh independent confirmation.',
 'R13':'W13 full type roster and softmax/lapse dynamics retained. Each positive shift record independently misleading with probability m; negative restoration changes trust only after the choice. Case check enumerates the positive-probability harmful branch at every positive rate. Finite trajectories with no harm remain in science; they are not redrawn.'}

def grid(**factors):return [dict(zip(factors,v)) for v in product(*factors.values())]

def settings(f):
    if f=='H1':return grid(overlap=['none','partial','heavy'])
    if f=='H2':return grid(d=[.3,.6],q=[0.,.2,.5])
    if f=='H3':return grid(r=[.3,.6,.9])
    if f=='E1':return grid(eta=[.005,.02,.1],theta=[.6,.8],environment=['stationary','forced','devaluation','shift','trade'])
    if f=='E2':return grid(kappa=[2.,8.],overlap=[6,4,3])
    if f=='E3':return grid(overlap=['partial','heavy'],rate=[.1,.3],truth=['amplifier','cached','innate'])
    if f=='U1':return grid(s=[.5,1.],tau=[.5,.8,1.])
    if f=='R1':return grid(beta=[1,3])
    if f=='R4':return [dict(alpha=10,placebo=False)]
    if f=='R13':return grid(mislead=[.1,.25,.5],learning_rate=[0.,.05,.2])
    raise ValueError(f)

def primary_setting(f):
    return dict(H1=dict(overlap='partial'),H2=dict(d=.6,q=.2),H3=dict(r=.6),E1=dict(eta=.1,theta=.6,environment='stationary'),E2=dict(kappa=2.,overlap=4),E3=dict(overlap='partial',rate=.1,truth='cached'),U1=dict(s=.5,tau=1.),R1=dict(beta=3),R4=dict(alpha=10,placebo=False),R13=dict(mislead=.25,learning_rate=.2))[f]

def texture_variants(f):
    # Finite one-factor variants in declared order. No selection from outcomes.
    overrides={
      'H1':[dict(tops=4),dict(tops=10),dict(pieces=8),dict(pieces=20),dict(max_piece_length=2),dict(max_piece_length=6),dict(episodes=3),dict(episodes=12),dict(mixed_tops=True)],
      'H2':[dict(units=16),dict(units=64),dict(hard=1),dict(hard=3),dict(low=.8,high=1.25),dict(low=.5,high=2.),dict(dominant=True)],
      'H3':[dict(units=12),dict(units=48),dict(discovery_fraction=.25),dict(discovery_fraction=.75),dict(partial=.25),dict(partial=.5),dict(partial=.75)],
      'E1':[dict(eta=.0025),dict(eta=.05),dict(theta=.7),dict(options=5),dict(dimensions=4),dict(environment='forced',forced_every=16),dict(environment='forced',forced_every=4),dict(environment='devaluation',devaluation=.25)],
      'E2':[dict(kappa=.5),dict(kappa=1.),dict(kappa=4.),dict(kappa=16.),dict(ends=4),dict(missing_piece=True)],
      'E3':[dict(rate=.05),dict(rate=.2),dict(rate=.5),dict(innate_share=.25),dict(innate_share=.5),dict(innate_share=.75),dict(fear=True)],
      'U1':[dict(tau=.25),dict(tau=.65),dict(s=.25),dict(s=.75),dict(claims=16),dict(claims=64),dict(claims=128)],
      'R1':[dict(beta=2),dict(beta=4),dict(structure='constant'),dict(structure='varying'),dict(structure='graded')],
    }.get(f,[])
    return [dict(primary_setting(f),**{})|x for x in overrides]

def selected_settings(f,shed=0):
    rows=settings(f)
    if f=='H1':
        if shed>=1:rows=[dict(s,prefixes=[1,2,4,8]) for s in rows]
        if shed>=2:rows=[dict(s,omit_pieces=True) for s in rows]
        if shed>=3:rows=[dict(s,prefixes=[4]) if s['overlap']=='heavy' else s for s in rows]
    if f=='H2':
        if shed>=1:rows=[s for s in rows if s['d']==.6]
        if shed>=2:rows=[dict(s,omit_replacements=True) for s in rows]
    if f=='H3':
        if shed>=1:rows=[s for s in rows if s['r']!=.3]
        if shed>=2:rows=[dict(s,omit_discovery=True) for s in rows]
    if f=='E1':
        if shed>=1:rows=[s for s in rows if s['theta']!=.8]
        if shed>=2:rows=[s for s in rows if s['environment']!='shift']
        if shed>=3:rows=[s for s in rows if s['eta']!=.02]
    if f=='E2':
        if shed>=1:rows=[s for s in rows if s['kappa']!=8]
        if shed>=2:rows=[s for s in rows if s['overlap']!=3]
    if f=='E3' and shed>=1:rows=[s for s in rows if s['truth']!='innate']
    if f=='U1' and shed>=1:rows=[dict(s,omit_menu=True) for s in rows]
    if f=='R1':
        if shed>=1:rows=[s for s in rows if s['beta']!=1]
        if shed>=2:rows=[dict(s,training_count=2) for s in rows]
    if f=='R4':
        if shed>=1:rows=[dict(s,omit_exact_mean=True) for s in rows]
        if shed>=2:rows=[dict(s,prefixes=[128,256,512,1024]) for s in rows]
    if f=='R13' and shed>=1:rows=[s for s in rows if s['mislead']!=.1]
    return rows

def card(f):
    return dict(id=f,name=NAMES[f],ceiling_seconds=FAMILY_HOURS[f]*3600,blocks=16,settings=settings(f),
        choices=CHOICES[f],routes=ROUTES[f],texture_variants=texture_variants(f),independent_unit='whole-world block',
        architecture='miniature, architecture untested',inference='descriptive, no confirmation or human inference',
        practical_bar=.05 if f not in ('E1','E2','U1','R13') else 'one sampled record level or E2 distance bin',
        provisional_choices=True)
