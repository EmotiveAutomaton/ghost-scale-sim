"""Prospective November 4 exploratory cards and supplement-aware admission.

Prior science is immutable. These cards are new descriptive maps, not confirmation.
"""
import itertools
import math
from datetime import datetime, timedelta
import numpy as np
from . import contract, registry
from .common import rng, posterior, close


def crossing_metrics(path):
    """One-nat crossing, recrossing and finite-horizon persistence; no censor imputation."""
    x=[float(v) for v in path]
    if not x or not all(math.isfinite(v) for v in x):raise ValueError('finite nonempty evidence path required')
    hits=[i+1 for i,v in enumerate(x) if v>=1.]
    last_below=max((i+1 for i,v in enumerate(x) if v<1.),default=0)
    return dict(first_crossing=hits[0] if hits else None,
                sustained_crossing=last_below+1 if last_below<len(x) else None,
                ever_crossed=float(bool(hits)),final_above=float(x[-1]>=1.),
                recrossed_below=float(bool(hits) and any(v<1 for v in x[hits[0]:])),
                fraction_above=sum(v>=1 for v in x)/len(x),final_log_odds=x[-1])


def cards():
    result=[]
    def add(f, settings, blocks, question, case):
        i=1+sum(c['family']==f for c in result)
        result.append(dict(id=f'{f}-X{i:02d}',family=f,settings=settings,blocks=blocks,
            question=question,case=case,precision=f'{blocks} independent worlds per cell; maximum binomial Monte Carlo standard error {0.5/math.sqrt(blocks):.4f}; log-score uncertainty estimated across worlds.',
            interpretation='Exploratory constructed-method map; miniature, architecture untested; no confirmation or human inference.',
            routes=list(registry.ROUTES[f]),seed_domain=f'v26-friday-20261104-{f.lower()}-x{i:02d}',
            probe_rule='One excluded world at every declared setting; complete original, replay and scalar reconstruction. Per-pass cost uses the largest observed whole-world time with 50% margin, plus overhead.',
            controls='Inherited exact-law known answers plus Friday-specific alias, persistence, source and setting checks. Controls never impose a desired science effect.',
            construction='Makers are nested within worlds. Reader comparisons share observations within a setting. Different setting cells use independent worlds; no paired causal claim. All unavailable outputs remain explicit.'))
    for tops,pieces in itertools.product((4,6,10),(8,12,20)):
        add('H1',[dict(tops=tops,pieces=pieces,episodes=ep,overlap=o,prefixes=[1,2,4,8,16]) for ep,o in itertools.product((3,6,12),('partial','heavy'))],64,
            f'How does library overlap interact with episode length at {tops} top goals and {pieces} pieces?',
            'At least 25% of excluded worlds remain uncertain about the true top goal after one document, at every setting.')
    for eta,share in itertools.product((.0025,.005,.02),(.5,.75,1.)):
        horizon=2048 if eta<.02 else 1024
        environments=['stationary','forced','trade']+(['shift'] if share<1 else [])
        add('E1',[dict(eta=eta,theta=.6,specialists=share,environment=e,max_history=horizon,
                       histories=[64,128,256,512,1024]+([2048] if horizon==2048 else []),truth_types=['A','B'],evidence_path=True) for e in environments],64,
            f'At learning rate {eta} and specialist share {share}, do one-nat crossings persist, and how often do fixed-value makers cross falsely?',
            'Every menu condition has differing fixed/circulating policies after an informative outcome in at least 25% of deterministic grid fixtures. Shift is tested only where it changes menus; threshold theta is not a B-truth intervention.')
    for d in (.3,.6):
        add('H2',[dict(d=d,q=q,units=n,hard=h) for q,n,h in itertools.product((.2,.5),(16,32,64),(1,3))],64,
            f'With deliberation probability {d}, how do observation dose and difficulty burden change the value and misreading of pauses?',
            'At least 25% of excluded worlds remain uncertain under text-only reading at each setting; both deliberation and difficulty have positive probability.')
    for rate in (.3,.6,.9):
        add('H3',[dict(r=rate,partial=p,discovery_fraction=f,units=n) for p,f,n in itertools.product((0.,.5,.75),(.25,.75),(12,48))],64,
            f'With revision rate {rate}, how do incomplete revision and discovery timing change initial versus settled-goal recovery?',
            'Changed-goal cases have positive probability .5; at least 25% of excluded changed-goal documents remain uncertain under the final-text reader at each setting.')
    for ends,missing in itertools.product((3,4),(False,True)):
        add('E2',[dict(kappa=k,overlap=o,ends=ends,missing_piece=missing) for k,o in itertools.product((.5,2.,8.,16.),(3,4,6))],128,
            f'With {ends} ends and missing-piece proxy {missing}, where do own, population and mixture priors cross as prior concentration and library mismatch vary?',
            'Every finite maker weight has candidates in all four declared distance bins; bin assignment is balanced by construction.')
    for overlap,fear in itertools.product(('partial','heavy'),(False,True)):
        add('E3',[dict(overlap=overlap,fear=fear,rate=rate,innate_share=mix,truth='cached') for rate,mix in itertools.product((.05,.1,.3,.5),(0.,.25,.75))],64,
            f'Under {overlap} library overlap and fear channel {fear}, when do learning rate and innate mixing change amplifier-reader harm?',
            'At least 25% of excluded eligible observations give amplifier and cached signature probabilities differing by .2, at every setting.')
    for strength in (.25,.5,.75,1.):
        add('U1',[dict(s=strength,tau=1.,claims=n,training_lengths=list(range(1,65))) for n in (16,32,64,128)],128,
            f'At source strength {strength}, how does new-claim dose change the source-history learning curve and its uncertainty?',
            'Enumerated revealed pools distinguish the generating source from unbiased selection by more than .1 with probability at least .25 under each hypothesis. Strengths .25/.75 are deliberately outside the reader grid, not exact-model consistency tests.')
    for structure in ('constant','varying'):
        add('R1',[dict(beta=b,structure=structure,switch_gap=g) for b,g in itertools.product((1,3),(0.,.125,.25,.5,1.))],64,
            f'Under {structure} passive effects, how does actual intervention strength alter split-alias recovery?',
            'Passive split aliases hold exactly at every setting; positive-gap switch interventions separate every split pair, and zero gap is an exact negative control. Remove cannot split the retained pair; constant-effect satiation remains aliased.')
    return result


def refill_cards():
    """Prospectively declared work for the first genuinely drained Friday queue.

    The E2/U1 successors replace preparations held before admission; they do not
    repeat a scientific draw. H1-X10 is a cost-driven subset of the unavailable
    ten-goal grid. The H2/H3 cards fill low-rate boundaries left by the completed
    interior maps.
    """
    original={card['id']:card for card in cards()}
    result=[]

    for old_id,new_id in (
        ('E2-X02','E2-X02R1'),('E2-X03','E2-X03R1'),('E2-X04','E2-X04R1'),
        ('U1-X03','U1-X03R1'),('U1-X04','U1-X04R1'),
    ):
        card=dict(original[old_id])
        card.update(
            id=new_id,
            seed_domain=f'v26-friday-20261104-{new_id.lower()}',
            resource_successor_of=old_id,
            held_packet=f'v26-friday-20261104-{old_id.lower()}-probe',
            registration_reason=(
                'The predecessor preparation was held before admission after a '
                'separate resource-estimation failure. Its verification-only '
                'healing is complete; this new source-bound plan repeats no old '
                'scientific draw.'
            ),
        )
        result.append(card)

    def add(card_id,family,settings,blocks,question,case,reason):
        result.append(dict(
            id=card_id,family=family,settings=settings,blocks=blocks,
            question=question,case=case,
            precision=f'{blocks} independent worlds per cell; maximum binomial Monte Carlo standard error {0.5/math.sqrt(blocks):.4f}; log-score uncertainty estimated across worlds.',
            interpretation='Exploratory constructed-method map; miniature, architecture untested; no confirmation or human inference.',
            routes=list(registry.ROUTES[family]),
            seed_domain=f'v26-friday-20261104-{card_id.lower()}',
            probe_rule='One excluded world at every declared setting; complete original, replay and scalar reconstruction. Per-pass cost uses the largest observed whole-world time with 50% margin, plus overhead.',
            controls='Inherited exact-law known answers plus Friday-specific alias, persistence, source and setting checks. Controls never impose a desired science effect.',
            construction='Makers are nested within worlds. Reader comparisons share observations within a setting. Different setting cells use independent worlds; no paired causal claim. All unavailable outputs remain explicit.',
            registration_reason=reason,
        ))

    add(
        'H1-X10','H1',
        [dict(tops=10,pieces=12,episodes=ep,overlap=o,prefixes=[1,2,4,8,16])
         for ep,o in itertools.product((3,12),('partial','heavy'))],
        48,
        'At 10 top goals and 12 pieces, does the short-versus-long episode contrast persist under partial and heavy overlap?',
        'At least 25% of excluded worlds remain uncertain about the true top goal after one document, at every setting.',
        'The six-cell 64-world H1-X08 study was never admitted because its measured reservation exceeded the family ceiling. This prospectively frozen four-cell, 48-world map isolates the episode endpoints and both overlap conditions with a declared maximum binomial Monte Carlo standard error of 0.0722.',
    )
    add(
        'H2-X03','H2',
        [dict(d=.1,q=q,units=n,hard=h) for q,n,h in itertools.product((.2,.5),(16,32,64),(1,3))],
        64,
        'With deliberation probability 0.1, how do observation dose and difficulty burden change the value and misreading of pauses?',
        'At least 25% of excluded worlds remain uncertain under text-only reading at each setting; both deliberation and difficulty have positive probability.',
        'The completed .3 and .6 maps leave the low-deliberation boundary unresolved; this card freezes the same dose and difficulty grid at 0.1 before any new data.',
    )
    add(
        'H3-X04','H3',
        [dict(r=.1,partial=p,discovery_fraction=f,units=n) for p,f,n in itertools.product((0.,.5,.75),(.25,.75),(12,48))],
        64,
        'With revision rate 0.1, how do incomplete revision and discovery timing change initial versus settled-goal recovery?',
        'Changed-goal cases have positive probability .5; at least 25% of excluded changed-goal documents remain uncertain under the final-text reader at each setting.',
        'The completed .3, .6 and .9 maps leave the low-revision boundary unresolved; this card freezes the same completeness, timing and document-length grid at 0.1 before any new data.',
    )
    return result


def continuation_refill_cards():
    """Two first-generation evidence-dose maps after the first refill drained."""
    result=[]
    for card_id,training_count in (('R1-X03',2),('R1-X04',1)):
        settings=[
            dict(beta=beta,structure=structure,switch_gap=gap,training_count=training_count)
            for structure,beta,gap in itertools.product(
                ('constant','varying'),(1,3),(0.,.125,.25,.5,1.)
            )
        ]
        result.append(dict(
            id=card_id,family='R1',settings=settings,blocks=64,
            question=(
                f'With {training_count} distinct passive training menu'
                f'{"s" if training_count != 1 else ""}, how do evidence diversity, '
                'effect structure and actual intervention strength alter split-alias recovery?'
            ),
            case=(
                'Every retained passive menu preserves each declared split alias exactly; '
                'positive-gap switch interventions separate every split pair, zero gap is '
                'an exact negative control, remove cannot split the retained pair, and '
                'constant-effect satiation remains aliased.'
            ),
            precision=(
                '64 independent worlds per cell; maximum binomial Monte Carlo standard '
                'error 0.0625; log-score uncertainty estimated across worlds.'
            ),
            interpretation=(
                'Exploratory constructed-method evidence-dose map; miniature, architecture '
                'untested; no confirmation or human inference.'
            ),
            routes=list(registry.ROUTES['R1']),
            seed_domain=f'v26-friday-20261104-{card_id.lower()}',
            probe_rule=(
                'One excluded world at every declared setting; complete original, replay '
                'and scalar reconstruction. Per-pass cost uses the largest observed '
                'whole-world time with 50% margin, plus overhead.'
            ),
            controls=(
                'Inherited exact-law known answers plus Friday-specific passive-alias, '
                'zero-gap and positive-switch checks. Controls never impose a desired '
                'science effect.'
            ),
            construction=(
                'Makers are nested within worlds. Reader comparisons share observations '
                'within a setting. Different setting cells use independent worlds; no '
                'paired causal claim. All unavailable outputs remain explicit.'
            ),
            registration_reason=(
                'The completed R1-X01/X02 maps held passive evidence diversity at four '
                'menus. This first-generation evidence-dose card prospectively reduces '
                f'that diversity to {training_count} while retaining both effect structures, '
                'both precisions and all five frozen intervention gaps.'
            ),
        ))
    return result


def final_refill_cards():
    """Two narrow hierarchy endpoint maps after the second refill drained.

    H1-X10 found that the short-versus-long episode contrast changed sign at
    ten top goals and twelve pieces.  The original six-cell maps at eight and
    twenty pieces were cost-held before admission.  These cards retain only
    the two episode endpoints and both overlap conditions, and reduce precision
    prospectively so both endpoint checks can fit the remaining H1 ceiling.
    """
    result=[]
    for card_id,pieces,held in (
        ('H1-X11',8,'H1-X07'),
        ('H1-X12',20,'H1-X09'),
    ):
        blocks=32
        result.append(dict(
            id=card_id,family='H1',
            settings=[
                dict(tops=10,pieces=pieces,episodes=episodes,overlap=overlap,
                     prefixes=[1,2,4,8,16])
                for episodes,overlap in itertools.product((3,12),('partial','heavy'))
            ],
            blocks=blocks,
            question=(
                f'At 10 top goals and {pieces} pieces, does the short-versus-long '
                'episode contrast retain the sign pattern observed at 12 pieces '
                'under partial and heavy overlap?'
            ),
            case=(
                'At least 25% of excluded worlds remain uncertain about the true '
                'top goal after one document, at every setting.'
            ),
            precision=(
                f'{blocks} independent worlds per cell; maximum binomial Monte '
                f'Carlo standard error {0.5/math.sqrt(blocks):.4f}; log-score '
                'uncertainty estimated across worlds.'
            ),
            interpretation=(
                'Exploratory constructed-method hierarchy endpoint map; miniature, '
                'architecture untested; no confirmation or human inference.'
            ),
            routes=list(registry.ROUTES['H1']),
            seed_domain=f'v26-friday-20261104-{card_id.lower()}',
            probe_rule=(
                'One excluded world at every declared setting; complete original, '
                'replay and scalar reconstruction. Per-pass cost uses the largest '
                'observed whole-world time with 50% margin, plus overhead.'
            ),
            controls=(
                'Inherited exact-law known answers plus Friday-specific alias, '
                'persistence, source and setting checks. Controls never impose a '
                'desired science effect.'
            ),
            construction=(
                'Makers are nested within worlds. Reader comparisons share '
                'observations within a setting. Different setting cells use '
                'independent worlds; no paired causal claim. All unavailable '
                'outputs remain explicit.'
            ),
            held_predecessor=held,
            registration_reason=(
                f'The six-cell 64-world {held} study was held before admission '
                'because its measured reservation exceeded the then-available H1 '
                'ceiling. After H1-X10 exposed a sign change between episode '
                'endpoints at 12 pieces, this new card prospectively tests the '
                f'{pieces}-piece endpoint with four cells and 32 worlds per cell. '
                'It is a lower-precision exploratory map, not a retry or confirmation.'
            ),
        ))
    return result


def rare_event_refill_cards():
    """Rare-deliberation and rare-revision maps after the endpoint refill drained.

    The completed 0.1 maps show larger, noisier information per event than the
    0.3--0.9 maps.  These cards freeze one lower rate before new data.  Their
    doubled whole-world count is a prospective precision correction for rarer
    events, not an extension selected after looking at the new outcomes.
    """
    blocks=128
    common=dict(
        blocks=blocks,
        precision=(
            f'{blocks} independent worlds per cell; maximum binomial Monte Carlo '
            f'standard error {0.5/math.sqrt(blocks):.4f}; log-score uncertainty '
            'estimated across worlds. The larger whole-world count is fixed in '
            'advance because the target events occur with probability 0.05.'
        ),
        interpretation=(
            'Exploratory constructed-method rare-event boundary map; miniature, '
            'architecture untested; no confirmation or human inference.'
        ),
        probe_rule=(
            'One excluded world at every declared setting; complete original, '
            'replay and scalar reconstruction. Per-pass cost uses the largest '
            'observed whole-world time with 50% margin, plus overhead.'
        ),
        controls=(
            'Inherited exact-law known answers plus Friday-specific alias, '
            'persistence, source and setting checks. Controls never impose a '
            'desired science effect.'
        ),
        construction=(
            'Makers are nested within worlds. Reader comparisons share '
            'observations within a setting. Different setting cells use '
            'independent worlds; no paired causal claim. All unavailable '
            'outputs remain explicit.'
        ),
    )
    h2=dict(
        common,id='H2-X04',family='H2',
        settings=[dict(d=.05,q=q,units=n,hard=h)
                  for q,n,h in itertools.product((.2,.5),(16,32,64),(1,3))],
        question=(
            'With deliberation probability 0.05, does the higher and noisier '
            'per-deliberation value seen at 0.1 persist across observation dose '
            'and difficulty burden?'
        ),
        case=(
            'At least 25% of excluded worlds remain uncertain under text-only '
            'reading at each setting; both deliberation and difficulty have '
            'positive probability.'
        ),
        routes=list(registry.ROUTES['H2']),
        seed_domain='v26-friday-20261104-h2-x04',
        registration_reason=(
            'The completed 0.1 pause map has larger and noisier information per '
            'deliberation than the 0.3 and 0.6 maps. This card freezes the same '
            'dose and difficulty grid at 0.05 with 128 worlds per cell before '
            'new data, to test the rare-deliberation boundary without calling it '
            'confirmation.'
        ),
    )
    h3=dict(
        common,id='H3-X05',family='H3',
        settings=[dict(r=.05,partial=p,discovery_fraction=f,units=n)
                  for p,f,n in itertools.product((0.,.5,.75),(.25,.75),(12,48))],
        question=(
            'With revision rate 0.05, does the larger per-revision value seen at '
            '0.1 persist across revision completeness, discovery timing and '
            'document length?'
        ),
        case=(
            'Changed-goal cases have positive probability .5; at least 25% of '
            'excluded changed-goal documents remain uncertain under the final-text '
            'reader at each setting.'
        ),
        routes=list(registry.ROUTES['H3']),
        seed_domain='v26-friday-20261104-h3-x05',
        registration_reason=(
            'The completed 0.1 revision map has larger per-revision values and as '
            'few as 48 available worlds in a cell. This card freezes the same '
            'completeness, timing and length grid at 0.05 with 128 worlds per cell '
            'before new data, prospectively compensating for rarer revisions.'
        ),
    )
    return [h2,h3]


def lower_rate_refill_cards():
    """Prospective lower-rate pause and revision maps after the 0.05 maps.

    H2-X04 remains positive across its complete 0.05 grid, so H2-X05 tests one
    lower deliberation rate without selecting cells.  H3-X05 crosses zero across
    its 0.05 cells.  H3-X06 therefore retains the declared completeness
    endpoints, both discovery times and both document lengths at one lower rate;
    it does not select cells by their observed sign.
    """
    blocks=256
    common=dict(
        blocks=blocks,
        precision=(
            f'{blocks} independent worlds per cell; maximum binomial Monte Carlo '
            f'standard error {0.5/math.sqrt(blocks):.4f}; log-score uncertainty '
            'estimated across worlds. The whole-world count is doubled before '
            'new data because the target event rate is halved from 0.05 to 0.025.'
        ),
        interpretation=(
            'Exploratory constructed-method lower-rate boundary map; miniature, '
            'architecture untested; no confirmation or human inference.'
        ),
        probe_rule=(
            'One excluded world at every declared setting; complete original, '
            'replay and scalar reconstruction. Per-pass cost uses the largest '
            'observed whole-world time with 50% margin, plus overhead.'
        ),
        controls=(
            'Inherited exact-law known answers plus Friday-specific alias, '
            'persistence, source and setting checks. Controls never impose a '
            'desired science effect.'
        ),
        construction=(
            'Makers are nested within worlds. Reader comparisons share '
            'observations within a setting. Different setting cells use '
            'independent worlds; no paired causal claim. All unavailable '
            'outputs remain explicit.'
        ),
    )
    h2=dict(
        common,id='H2-X05',family='H2',
        settings=[dict(d=.025,q=q,units=n,hard=h)
                  for q,n,h in itertools.product((.2,.5),(16,32,64),(1,3))],
        question=(
            'With deliberation probability 0.025, does the positive but noisy '
            'per-deliberation value at 0.05 persist across the complete '
            'observation-dose and difficulty grid?'
        ),
        case=(
            'At least 25% of excluded worlds remain uncertain under text-only '
            'reading at each setting; both deliberation and difficulty have '
            'positive probability.'
        ),
        routes=list(registry.ROUTES['H2']),
        seed_domain='v26-friday-20261104-h2-x05',
        registration_reason=(
            'All 12 completed 0.05 pause cells have positive per-deliberation '
            'means, while their magnitude remains noisy and nonmonotonic across '
            'dose and difficulty. This card freezes the unchanged grid at 0.025 '
            'with 256 worlds per cell before new data. It is a separate lower-rate '
            'boundary map, not confirmation or an extension of the prior draw.'
        ),
    )
    h3=dict(
        common,id='H3-X06',family='H3',
        settings=[dict(r=.025,partial=p,discovery_fraction=f,units=n)
                  for p,f,n in itertools.product((0.,.75),(.25,.75),(12,48))],
        question=(
            'With revision rate 0.025, how does the cell-to-cell sign variation '
            'seen at 0.05 behave at the complete and mostly incomplete revision '
            'endpoints across discovery timing and document length?'
        ),
        case=(
            'Changed-goal cases have positive probability .5; at least 25% of '
            'excluded changed-goal documents remain uncertain under the final-text '
            'reader at each setting.'
        ),
        routes=list(registry.ROUTES['H3']),
        seed_domain='v26-friday-20261104-h3-x06',
        registration_reason=(
            'The completed 0.05 revision map crosses zero across independent '
            'cells. A full doubled 12-cell map would press against the remaining '
            'H3 family ceiling, so this card prospectively retains the natural '
            'completeness endpoints, both discovery times and both document '
            'lengths at 0.025. It selects no cell by observed sign and is not '
            'confirmation or a paired contrast.'
        ),
    )
    return [h2,h3]


def revision_length_refill_cards():
    """Prospective length-transition maps after the 0.025 revision endpoints.

    H3-X06 changed from generally positive 12-unit cells to negative 48-unit
    cells.  These two sibling cards freeze intermediate lengths separately for
    complete and mostly incomplete revisions.  They retain both discovery times
    and do not choose a single observed cell or lower the revision rate again.
    """
    blocks=256
    common=dict(
        blocks=blocks,
        precision=(
            f'{blocks} independent worlds per cell; maximum binomial Monte Carlo '
            f'standard error {0.5/math.sqrt(blocks):.4f}; log-score uncertainty '
            'estimated across worlds. The rare-event whole-world count is retained '
            'from H3-X06 before new data.'
        ),
        interpretation=(
            'Exploratory constructed-method revision-length localization; '
            'miniature, architecture untested; no confirmation or human inference.'
        ),
        probe_rule=(
            'One excluded world at every declared setting; complete original, '
            'replay and scalar reconstruction. Per-pass cost uses the largest '
            'observed whole-world time with 50% margin, plus overhead.'
        ),
        controls=(
            'Inherited exact-law known answers plus Friday-specific alias, '
            'persistence, source and setting checks. Controls never impose a '
            'desired science effect.'
        ),
        construction=(
            'Makers are nested within worlds. Reader comparisons share '
            'observations within a setting. Different setting cells use '
            'independent worlds; no paired causal claim. All unavailable '
            'outputs remain explicit.'
        ),
        case=(
            'Changed-goal cases have positive probability .5; at least 25% of '
            'excluded changed-goal documents remain uncertain under the final-text '
            'reader at each setting.'
        ),
        routes=list(registry.ROUTES['H3']),
    )
    complete=dict(
        common,id='H3-X07',family='H3',
        settings=[dict(r=.025,partial=0.,discovery_fraction=f,units=n)
                  for f,n in itertools.product((.25,.75),(24,36))],
        question=(
            'At revision rate 0.025 with complete revisions, where between 12 '
            'and 48 document units does the positive-to-negative per-revision '
            'transition occur across early and late discovery?'
        ),
        seed_domain='v26-friday-20261104-h3-x07',
        registration_reason=(
            'Both complete-revision H3-X06 cells are positive at 12 units and '
            'negative at 48 units. This card freezes 24 and 36 units at both '
            'declared discovery times before new data, localizing rather than '
            'confirming the independent-cell sign transition.'
        ),
    )
    incomplete=dict(
        common,id='H3-X08',family='H3',
        settings=[dict(r=.025,partial=.75,discovery_fraction=f,units=n)
                  for f,n in itertools.product((.25,.75),(24,36))],
        question=(
            'At revision rate 0.025 with mostly incomplete revisions, how do '
            'intermediate document lengths and discovery timing organize the '
            'mixed short-document and negative long-document cells?'
        ),
        seed_domain='v26-friday-20261104-h3-x08',
        registration_reason=(
            'Mostly incomplete H3-X06 cells are negative at 48 units, while the '
            '12-unit cells differ by discovery time. This card freezes 24 and '
            '36 units at both discovery times before new data, retaining the '
            'whole interaction instead of selecting one observed cell.'
        ),
    )
    return [complete,incomplete]


def midpoint_and_half_rate_refill_cards():
    """Prospective H3 midpoint and H2 half-rate maps after H3-X07/X08.

    Both H3 length-localization maps are negative by 24 units, while the
    complete 12-unit endpoints were positive and the mostly incomplete
    endpoints differed by discovery time.  H3-X09 freezes the midpoint of the
    remaining 12--24 interval across the whole completeness/discovery
    interaction.  H2-X06 halves the deliberation rate once more while retaining
    its complete grid; its world count doubles prospectively for the rarer
    event.  Neither card selects a setting by its new outcome.
    """
    common=dict(
        probe_rule=(
            'One excluded world at every declared setting; complete original, '
            'replay and scalar reconstruction. Per-pass cost uses the largest '
            'observed whole-world time with 50% margin, plus overhead.'
        ),
        controls=(
            'Inherited exact-law known answers plus Friday-specific alias, '
            'persistence, source and setting checks. Controls never impose a '
            'desired science effect.'
        ),
        construction=(
            'Makers are nested within worlds. Reader comparisons share '
            'observations within a setting. Different setting cells use '
            'independent worlds; no paired causal claim. All unavailable '
            'outputs remain explicit.'
        ),
    )
    h2_blocks=512
    h2=dict(
        common,id='H2-X06',family='H2',blocks=h2_blocks,
        settings=[dict(d=.0125,q=q,units=n,hard=h)
                  for q,n,h in itertools.product((.2,.5),(16,32,64),(1,3))],
        question=(
            'With deliberation probability 0.0125, does the positive '
            'per-deliberation value at 0.025 persist across the complete '
            'observation-dose and difficulty grid?'
        ),
        case=(
            'At least 25% of excluded worlds remain uncertain under text-only '
            'reading at each setting; both deliberation and difficulty have '
            'positive probability.'
        ),
        precision=(
            f'{h2_blocks} independent worlds per cell; maximum binomial Monte '
            f'Carlo standard error {0.5/math.sqrt(h2_blocks):.4f}; log-score '
            'uncertainty estimated across worlds. The whole-world count is '
            'doubled before new data because the target event rate is halved '
            'from 0.025 to 0.0125.'
        ),
        interpretation=(
            'Exploratory constructed-method half-rate boundary map; miniature, '
            'architecture untested; no confirmation or human inference.'
        ),
        routes=list(registry.ROUTES['H2']),
        seed_domain='v26-friday-20261104-h2-x06',
        registration_reason=(
            'All 12 H2-X05 cells remain positive at deliberation probability '
            '0.025. This card freezes the unchanged grid at 0.0125 and doubles '
            'the whole-world count before new data. It tests a lower event-rate '
            'boundary without selecting one favorable cell or calling the '
            'earlier exploratory map confirmation.'
        ),
    )
    h3_blocks=224
    h3=dict(
        common,id='H3-X09',family='H3',blocks=h3_blocks,
        settings=[dict(r=.025,partial=p,discovery_fraction=f,units=18)
                  for p,f in itertools.product((0.,.75),(.25,.75))],
        question=(
            'At revision rate 0.025 and 18 document units, how do revision '
            'completeness and discovery timing organize the transition between '
            'the 12-unit endpoints and the negative 24-unit cells?'
        ),
        case=(
            'Changed-goal cases have positive probability .5; at least 25% of '
            'excluded changed-goal documents remain uncertain under the final-text '
            'reader at each setting.'
        ),
        precision=(
            f'{h3_blocks} independent worlds per cell; maximum binomial Monte '
            f'Carlo standard error {0.5/math.sqrt(h3_blocks):.4f}; log-score '
            'uncertainty estimated across worlds. The near-256-world count is '
            'fixed before new data to retain rare-event precision while leaving '
            'measured reservation headroom under the H3 family ceiling.'
        ),
        interpretation=(
            'Exploratory constructed-method revision-length midpoint map; '
            'miniature, architecture untested; no confirmation or human inference.'
        ),
        routes=list(registry.ROUTES['H3']),
        seed_domain='v26-friday-20261104-h3-x09',
        registration_reason=(
            'H3-X07 and H3-X08 are negative at both 24- and 36-unit lengths. '
            'The complete 12-unit H3-X06 endpoints were positive, while the '
            'mostly incomplete 12-unit endpoints differed by discovery time. '
            'This card freezes the arithmetic midpoint of the remaining 12-to-24 '
            'interval across both completeness endpoints and discovery times. '
            'It selects no newly observed cell and fits the remaining H3 family '
            'allowance prospectively.'
        ),
    )
    return [h2,h3]


def transition_and_quarter_rate_refill_cards():
    """Prospective H3 15-unit transition and H2 quarter-rate maps.

    H3-X09 leaves a heterogeneous four-cell transition at 18 units, after the
    corresponding 12-unit endpoints and before the uniformly negative 24-unit
    cells.  H3-X10 freezes the arithmetic midpoint of the remaining 12--18
    interval across the whole completeness/discovery interaction.  H2-X07
    halves the deliberation rate once more after every H2-X06 cell remained
    positive and doubles the whole-world count before data to retain rare-event
    precision.  Neither card selects a setting by its new outcome.
    """
    common=dict(
        probe_rule=(
            'One excluded world at every declared setting; complete original, '
            'replay and scalar reconstruction. Per-pass cost uses the largest '
            'observed whole-world time with 50% margin, plus overhead.'
        ),
        controls=(
            'Inherited exact-law known answers plus Friday-specific alias, '
            'persistence, source and setting checks. Controls never impose a '
            'desired science effect.'
        ),
        construction=(
            'Makers are nested within worlds. Reader comparisons share '
            'observations within a setting. Different setting cells use '
            'independent worlds; no paired causal claim. All unavailable '
            'outputs remain explicit.'
        ),
    )
    h2_blocks=1024
    h2=dict(
        common,id='H2-X07',family='H2',blocks=h2_blocks,
        settings=[dict(d=.00625,q=q,units=n,hard=h)
                  for q,n,h in itertools.product((.2,.5),(16,32,64),(1,3))],
        question=(
            'With deliberation probability 0.00625, does the positive '
            'per-deliberation value at 0.0125 persist across the complete '
            'observation-dose and difficulty grid?'
        ),
        case=(
            'At least 25% of excluded worlds remain uncertain under text-only '
            'reading at each setting; both deliberation and difficulty have '
            'positive probability.'
        ),
        precision=(
            f'{h2_blocks} independent worlds per cell; maximum binomial Monte '
            f'Carlo standard error {0.5/math.sqrt(h2_blocks):.4f}; log-score '
            'uncertainty estimated across worlds. The whole-world count is '
            'doubled before new data because the target event rate is halved '
            'from 0.0125 to 0.00625.'
        ),
        interpretation=(
            'Exploratory constructed-method quarter-rate boundary map; miniature, '
            'architecture untested; no confirmation or human inference.'
        ),
        routes=list(registry.ROUTES['H2']),
        seed_domain='v26-friday-20261104-h2-x07',
        registration_reason=(
            'All 12 H2-X06 cells remain positive at deliberation probability '
            '0.0125. This card freezes the unchanged grid at 0.00625 and '
            'doubles the whole-world count before new data. It tests one lower '
            'event-rate boundary without selecting a favorable cell or calling '
            'the earlier exploratory map confirmation.'
        ),
    )
    h3_blocks=224
    h3=dict(
        common,id='H3-X10',family='H3',blocks=h3_blocks,
        settings=[dict(r=.025,partial=p,discovery_fraction=f,units=15)
                  for p,f in itertools.product((0.,.75),(.25,.75))],
        question=(
            'At revision rate 0.025 and 15 document units, how do revision '
            'completeness and discovery timing organize the transition between '
            'the 12-unit endpoints and the heterogeneous 18-unit cells?'
        ),
        case=(
            'Changed-goal cases have positive probability .5; at least 25% of '
            'excluded changed-goal documents remain uncertain under the final-text '
            'reader at each setting.'
        ),
        precision=(
            f'{h3_blocks} independent worlds per cell; maximum binomial Monte '
            f'Carlo standard error {0.5/math.sqrt(h3_blocks):.4f}; log-score '
            'uncertainty estimated across worlds. The H3-X09 whole-world count '
            'is retained before new data so measured reservation, not outcome, '
            'determines whether the card fits the remaining family allowance.'
        ),
        interpretation=(
            'Exploratory constructed-method revision-length transition map; '
            'miniature, architecture untested; no confirmation or human inference.'
        ),
        routes=list(registry.ROUTES['H3']),
        seed_domain='v26-friday-20261104-h3-x10',
        registration_reason=(
            'At 18 units H3-X09 is positive only in the complete early-discovery '
            'cell, near zero in the mostly incomplete early-discovery cell, and '
            'negative in both late-discovery cells. The corresponding 12-unit '
            'endpoints are positive except for the mostly incomplete late cell. '
            'This card freezes the arithmetic midpoint of the remaining 12-to-18 '
            'interval across the full interaction before new data.'
        ),
    )
    return [h2,h3]


def design_check(card):
    """Excluded grid checks inspect laws/short fixtures, never cost or science draws."""
    from . import hierarchy, values, pauses, revision, self_model, cached, opacity, alias_repair
    f=card['family'];rows=[]
    for index,s in enumerate(card['settings']):
        shares=[];detail={}
        if f=='H1':
            check=hierarchy.design_check([s]);shares=[check['checks'][0]['uncertain_share']]
        elif f in ('H2','H3'):
            uncertain=0
            for b in range(16):
                ident=['friday-check-only',card['id'],index,b];r=rng(*ident);w=pauses.weights(ident)
                if f=='H2':
                    laws=np.array([pauses.law(h,g,s['d'],s['q'],tuple(range(s['hard']))).sum((1,2)) for g in w for h in pauses.HABITS]);ll=np.zeros(18)
                    for t in range(s['units']):
                        obs,_=pauses.observe(r,pauses.HABITS[b%3],w[b%6],s['d'],s['q'],tuple(range(s['hard'])));ll+=np.log(laws[:,obs[0]])
                    p=posterior(ll).reshape(6,3).sum(1);uncertain+=p[b%6]<.8
                else:
                    for attempt in range(1000):
                        obs=revision.draw(r,w,0,0,s['units'],s['r'],s['partial'],s['discovery_fraction'])
                        if obs['settled']!=0:break
                    else:raise ValueError('excluded changed-goal fixture exhausted')
                    hs=revision.hypotheses(s['units'],s['discovery_fraction']);prior=np.array([(.5 if a==g else .1)/108 for a,g,p,rt,h in hs])
                    p=posterior(revision.document_likelihood(w,obs,s['units'],'final',partial=s['partial'],fraction=s['discovery_fraction']),prior)
                    ps=np.bincount([h[1] for h in hs],weights=p,minlength=6);uncertain+=ps[obs['settled']]<.8
            shares=[uncertain/16]
        elif f=='E1':
            differences=0;changed=0
            for b in range(32):
                ident=['friday-check-only',card['id'],index,b];r=rng(*ident);menu,_,_,_=values.menu_at(r,129,s|dict(environment_dimension=0))
                hs=[('A',0,2,np.array([.25,.25,.5])),('B',s['eta'],2,np.array([.25,.25,.5]))];state=values.states(hs)
                values.update(hs,state,np.array([1.,0.,0.]),1)
                differences+=np.max(abs(values.policy(menu,hs,state)[0]-values.policy(menu,hs,state)[1]))>1e-8
                before=values.menu_at(rng(*ident),129,s|dict(environment='stationary',environment_dimension=0))[0]
                changed+=not np.array_equal(menu,before)
            shares=[differences/32];detail['changed_menu_share']=changed/32
            if s['environment']=='shift':shares.append(changed/32)
        elif f=='E2':
            w=values.grid(s['ends']);ok=[]
            for x in w:
                tv=np.abs(w-x).sum(1)/2
                ok.append(all(np.any(abs(tv-d)<1e-12) for d in (0,.25,.5)) and np.any(tv>=.75))
            shares=[sum(ok)/len(ok)];detail['all_bins_available']=all(ok)
        elif f=='E3':
            different=eligible=0;r=rng('friday-check-only',card['id'],index)
            for b in range(32):
                e=cached.library(s['overlap'],r);w=cached.simplex(4)[int(r.integers(15))];a=np.zeros(4)
                for t in range(128):
                    si=t%4;law=cached.softmax(3*w@e[list(cached.TRAIN[si])].T);piece=cached.TRAIN[si][int(r.choice(len(law),p=law))]
                    features=r.integers(0,2,4);ids=np.flatnonzero(features);blocked=r.random()<.02+.16*features[0]
                    available=r.random()<.02+.16*features[1] if s['fear'] else blocked
                    appraisal=(1-s['innate_share'])*(a[ids].mean() if len(ids) else 0)+s['innate_share']*features[0]
                    if available:eligible+=1;different+=abs(cached.fire(w@e[piece])-cached.fire(appraisal))>.2
                    a[ids]+=s['rate']*(int(blocked)-a[ids])
            shares=[different/eligible]
        elif f=='U1':
            for p in (.3,.7):
                shares.append(sum(math.comb(8,k)*p**k*(1-p)**(8-k) for k in range(9) if abs(opacity.conditional(k,s['s'])-opacity.conditional(k,0))>.1))
            detail['reader_misspecified']=s['s'] not in (0,.5,1.)
        elif f=='R1':
            hs=alias_repair.hypotheses(s['beta']);pairs=alias_repair.split_pairs(s['beta'],hs)
            passive=True;split=[]
            for b in range(4):
                e=alias_repair.library(rng('friday-check-only',card['id'],index,b),s['structure'],s['switch_gap'])
                for menu in alias_repair.TRAIN:
                    law=alias_repair.laws(e,menu,hs);passive &= all(np.max(abs(law[i]-law[j]))<1e-12 for i,j in pairs)
                laws=[alias_repair.laws(e,menu,hs) for menu in alias_repair.SWITCH]
                split.extend(any(np.max(abs(law[i]-law[j]))>1e-12 for law in laws) for i,j in pairs)
            expected=s['switch_gap']>0;shares=[float(passive),sum(v==expected for v in split)/len(split)];detail['switch_separation_share']=sum(split)/len(split)
        else:raise ValueError(f)
        rows.append(dict(setting=index,shares=shares,passed=all(v>=.25 for v in shares),**detail))
    return dict(passed=all(r['passed'] for r in rows),card=card['id'],seed_domain='friday-check-only; hierarchy uses check-only-H1',floor=.25,checks=rows,case=card['case'])


def admission(plan,accepted,supplement,current,charges,running=0):
    if accepted!=contract.acceptance(accepted['opened'],accepted['specification_sha256']):raise ValueError('unknown original acceptance')
    if current.tzinfo is None:raise ValueError('aware clock required')
    pools=supplement['pools_seconds']
    if sum(pools.values())!=110*3600 or supplement['total_cpu_seconds']!=110*3600:raise ValueError('unknown supplement total')
    if any(not math.isfinite(v) or v<0 for v in charges.values()):raise ValueError('invalid charges')
    if any(charges.get(k,0)>v for k,v in pools.items()):raise ValueError('existing pool overrun')
    if not datetime.fromisoformat(accepted['science_open'])<=current<min(datetime.fromisoformat(supplement['review_target']),datetime.fromisoformat(accepted['new_starts_close'])):raise ValueError('outside new admission window')
    if running>=2:raise ValueError('two shared scientific slots')
    if plan['numerical_threads']!=1 or plan['priority']!='below-normal':raise ValueError('resource mismatch')
    if plan['kind'] not in ('probe','study','healing'):raise ValueError('Friday exploratory packet mismatch')
    expected_pool='followup_healing' if plan['kind']=='healing' else 'studies'
    if plan['pool']!=expected_pool:raise ValueError('Friday exploratory pool mismatch')
    if plan['kind']=='healing' and plan.get('healing_number') not in (1,2):raise ValueError('healing limit')
    cap=plan['cpu_cap_seconds'];family=plan['family'];passes=plan['pass_caps']
    if not math.isfinite(cap) or cap<=0:raise ValueError('invalid cap')
    if set(passes)!={'original','replay','review'} or any(not math.isfinite(x) or x<=0 for x in passes.values()) or sum(passes.values())>cap:raise ValueError('pass reservation')
    if plan['kind']=='probe' and cap>1800:raise ValueError('probe ceiling')
    if charges.get(expected_pool,0)+cap>pools[expected_pool] or sum(charges.get(k,0) for k in pools)+cap>supplement['total_cpu_seconds']:raise ValueError('pool ceiling')
    if charges.get('family:'+family,0)+cap>supplement['family_seconds'][family]:raise ValueError('family ceiling')
    projection=plan['projected_wall_seconds']
    if not math.isfinite(projection) or projection<=0:raise ValueError('invalid wall projection')
    if current+timedelta(seconds=projection)>=datetime.fromisoformat(supplement['review_target']):raise ValueError('insufficient Friday review margin')
    if not plan.get('design_check_sha256') or not plan.get('supplement_sha256'):raise ValueError('missing prospective bindings')
    return True
