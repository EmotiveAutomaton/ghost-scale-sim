"""Post-hoc native acquisition diagnostic on retained binary provenance cases.

Reports recommend one of two public legal routines. The supplied task exposes
which one is useful to a full public-law checker. This deliberately preserves a
strong direct rival instead of making source inference necessary by construction.
"""
from . import world as W
from .verify import execute,code_cost


def apply(unit):
    truth=unit['evaluator']['truth'];w=W.make_world(0,80000+unit['index'])
    programs=[tuple(g) for g in w['groups']]
    target=sum(1<<v for v in (w['groups'][truth]+w['groups'][1-truth][:1]))
    checked=[W.enact(p,target) for p in programs]
    if not checked[truth]['success'] or checked[1-truth]['success']:raise ValueError('native diagnostic not separated')
    rows=[];tie_choices=[int(W.rng('source-uptake-tie',unit['mode'],unit['roots'],unit['index'],phase).integers(2)) for phase in range(2)]
    for row in unit['rows']:
        if row['instrument']!='valid':
            rows.append(dict(method=row['method'],valid=False));continue
        results=[]
        for phase,posterior in enumerate((row['posterior'],row['after_correction'][-1])):
            tied=abs(posterior[1]-posterior[0])<=1e-10
            selected=tie_choices[phase] if tied else int(posterior[1]>posterior[0]);result=dict(checked[selected]);result.update(selected_routine=selected,numerical_tie=tied)
            if execute(result['program'])!=result['artifact'] or code_cost(tuple(result['program']),list(map(tuple,result['library'])))!=result['code_cost']:raise ValueError('independent uptake execution failed')
            results.append(result)
        rows.append(dict(method=row['method'],valid=True,initial=results[0],corrected=results[1]))
    direct=max(range(2),key=lambda k:(checked[k]['success'],-checked[k]['code_cost'],-k))
    rows.append(dict(method='public-law-task-checker',valid=True,initial=dict(checked[direct],selected_routine=direct),corrected=dict(checked[direct],selected_routine=direct),
        counterfactual_check_work=sum(sum(c[k] for k in ('practice_actions','definition_actions','planning_evaluations','execution_actions')) for c in checked)))
    return dict(packet='C-'+unit['mode'],lineage=unit['index'],mode=unit['mode'],roots=unit['roots'],copies=unit['copies'],
        public=dict(programs=[list(p) for p in programs],target=target),evaluator=dict(recommendation_truth=truth),rows=rows,
        tie_policy=dict(tolerance=1e-10,choices=tie_choices,rule='same independent seeded coin for every method and descendant count within source lineage and stage'),
        scope='explicit bounded recommendation-to-practice policy; same retained source posteriors; public task law permits an inference-free stronger checker')
