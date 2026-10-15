"""Ghost transport/adapter; canonical models are loaded from ToMpathy unchanged."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
from .common import canonical, digest, write

TRANSPORT = 'ghost-jsonl-1'
BASE_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)


@lru_cache(None)
def schema_hash(schema):
    return schema.schema_hash()


def canonical_schema(directory, pin=None):
    directory = Path(directory)
    hashes = {name: digest(directory/name) for name in ('schema.py', 'ids.py')}
    if pin is not None and hashes != pin['files']:
        raise ValueError('canonical schema byte pin mismatch')
    package = '_ghost_canonical_'+hashes['schema.py'][:16]+'_'+hashes['ids.py'][:16]
    if package+'.schema' not in sys.modules:
        module = types.ModuleType(package); module.__path__ = [str(directory)]
        sys.modules[package] = module
        for name in ('ids', 'schema'):
            spec = importlib.util.spec_from_file_location(package+'.'+name, directory/(name+'.py'))
            item = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = item; spec.loader.exec_module(item)
    schema = sys.modules[package+'.schema']
    if pin is not None and (schema.SCHEMA_VERSION != pin['schema_version'] or
                            schema.schema_hash() != pin['schema_hash']):
        raise ValueError('canonical schema semantic pin mismatch')
    return schema, hashes


def _identity(*parts):
    return hashlib.sha256(canonical(['v23-ledger', *parts])).hexdigest()


class Emission:
    def __init__(self, schema, identity):
        self.schema, self.identity = schema, identity
        self.rows = {name: [] for name in schema.RECORDS}; self.blobs = {}
        self.maker, self.artifact, self.session = [_identity(identity, s) for s in ('maker', 'artifact', 'session')]
        self.add('Maker', maker_id=self.maker, source_dataset='ghost-v23-synthetic',
                 identity='synthetic', display_label='Synthetic maker')
        self.add('Artifact', artifact_id=self.artifact, maker_ids=[self.maker],
                 kind='other', source_uri='ghost://v23/'+_identity(identity), tier='T2', domain='constructed')
        self.add('Session', session_id=self.session, maker_id=self.maker,
                 artifact_id=self.artifact, t_start=BASE_TIME, t_end=BASE_TIME+timedelta(days=1))
        self.counter = 0

    def add(self, record_type, /, **fields):
        record = self.schema.RECORDS[record_type].model_validate(fields)
        row = record.model_dump(mode='json'); self.rows[record_type].append(row)
        return row

    def content(self, value):
        text = canonical(value).decode(); ref = hashlib.sha256(text.encode()).hexdigest()
        self.blobs[ref] = text; return ref

    def decision(self, channel, index, options, taken, *, observed=None, revision=False):
        self.counter += 1; ident = _identity(self.identity, 'decision', self.counter)
        moment = BASE_TIME+timedelta(seconds=self.counter)
        opts = [dict(option_id=_identity(ident, j), origin='maker', position=j,
                     content_ref=self.content(value), features={}) for j, value in enumerate(options)]
        row = self.add('Decision', decision_id=ident, maker_id=self.maker, session_id=self.session,
            t_open=moment, t_resolve=moment, options=opts,
            taken=opts[taken]['option_id'] if taken is not None else None,
            action='choose', context=dict(channel=channel, index=index, observed=observed or {}), outcome={})
        if revision and taken is not None:
            rev = self.add('Revision', revision_id=_identity(ident, 'revision'),
                artifact_id=self.artifact, maker_id=self.maker, t=moment,
                content_ref=opts[taken]['content_ref'])
            self.rows['Artifact'][0]['revisions'].append(rev['revision_id'])
        return row

    def finish(self):
        for kind in tuple(self.rows):
            if kind == 'Provenance':
                continue
            for row in self.rows[kind]:
                self.add('Provenance', record_id=row[self.schema.ID_FIELDS[kind]], record_type=kind,
                    adapter='ghost-v23-'+TRANSPORT, adapter_version='1',
                    source_file_sha256=hashlib.sha256(canonical(row)).hexdigest(), created_at=BASE_TIME)
        return dict(schema_version=self.schema.SCHEMA_VERSION, schema_hash=schema_hash(self.schema),
                    transport=TRANSPORT, rows=self.rows, blobs=self.blobs)


def emit(v, public, schema, identity):
    family = v['family']
    if family == 'N1': return emit_population(v, public, schema, identity)
    e = Emission(schema, identity)
    if family == 'R1':
        for name in ('own', 'claim'):
            e.decision(name, 0, [0, 1], public[name], revision=name == 'claim')
        for name in ('selection', 'operations'):
            for i, value in enumerate(public[name]):
                e.decision(name, i, [0, 1], value, revision=name == 'selection')
        e.decision('query', 0, [0, 1], None)
    elif family == 'R2':
        for i, (obs, flag) in enumerate(zip(public['observations'], public['flags'], strict=True)):
            e.decision('flag', i, [0, 1], flag)
            for h, value in enumerate(obs):
                if i >= 8 or h == 2:
                    e.decision('observation', i*3+h, [0, 1], value)
        e.decision('query', 0, [0, 1], None)
    elif family == 'R3':
        for i, row in enumerate(public['history']+[public['current']]):
            options = [dict(effect=a, cost=c) for a, c in zip(row['effects'], row['costs'], strict=True)]
            e.decision('history' if 'picked' in row else 'query', i, options, row.get('picked'), observed={'h': row['h']})
    elif family == 'R4':
        for i, value in enumerate(public['evidence']):
            if value >= 0:
                e.decision('leaf', i, [0, 1], value, revision=True)
        e.decision('cue', public['cue_node'], [0, 1], public['cue'], revision=True)
        e.decision('query', 0, [0, 1], None)
    elif family in ('N2', 'N5', 'N8'):
        classes = 4 if family == 'N5' else 2
        for i, row in enumerate(public['history']):
            observed = {k: value for k, value in row.items() if k != 'choice'}
            e.decision('history', i, list(range(classes)), row['choice'], observed=observed)
            if family == 'N8':
                e.add('ContextEvent', event_id=_identity(identity, 'threat', i), session_id=e.session,
                      t=BASE_TIME+timedelta(seconds=e.counter), kind='environment', payload={'threat': row['threat']})
        e.decision('query', 0, list(range(classes)), None,
                   observed=public.get('query', {}))
    elif family == 'N6':
        for i, row in enumerate(public['history']):
            for channel in ('claim', 'procedure'):
                e.decision(channel, i, [0, 1], row[channel])
        e.decision('query', 0, list(range(4)), None)
    elif family == 'N7':
        for i, value in enumerate(public['statement']):
            e.add('StatedPreference', pref_id=_identity(identity, 'statement', i), maker_id=e.maker,
                  attribute='recipe-score-'+str(i), value=value, scale=[], source='self_description', t=BASE_TIME)
        for i, row in enumerate(public['history']):
            e.decision('history', i, row['options'], row['choice'])
        e.decision('query', 0, public['query'], None)
    elif family == 'N4':
        if v['settings']['extension']:
            for i, row in enumerate(public['history']):
                e.decision('history', i, [0, 1], row[0], observed={'context': row[1]})
            e.decision('query', 0, [0, 1], None, observed={'changed': public['changed']})
            e.add('ContextEvent', event_id=_identity(identity, 'contingency-change'), session_id=e.session,
                  t=BASE_TIME+timedelta(seconds=e.counter), kind='environment', payload={'changed':public['changed']})
        else:
            for i, row in enumerate(public['history']):
                e.decision('history', i, row['options'], row['choice'],
                           observed=dict(need=row['need'], acted=row['acted']))
                e.add('ContextEvent', event_id=_identity(identity, 'need', i), session_id=e.session,
                      t=BASE_TIME+timedelta(seconds=e.counter), kind='environment', payload={'need': row['need']})
            e.decision('query', 0, public['query']['options'], None, observed={'need': public['query']['need']})
    elif family == 'N3':
        for i, row in enumerate(public['history']):
            d = e.decision('edit', i, [dict(delta=a, cost=c) for a, c in zip(row['edits'], row['costs'], strict=True)],
                row['picked'], observed={'x0': row['x0'], 'context': row['context']})
            e.add('ContextEvent', event_id=_identity(identity, 'edit-context', i), session_id=e.session,
                  t=BASE_TIME+timedelta(seconds=e.counter), kind='environment', payload={'local_context':row['context']})
            if row['picked']:
                after = (np.asarray(row['x0'])+np.asarray(row['edits'][row['picked']])).tolist()
                before_ref, after_ref = e.content(row['x0']), e.content(after)
                before_id, after_id = _identity(identity, 'before', i), _identity(identity, 'after', i)
                e.add('Revision', revision_id=before_id, artifact_id=e.artifact, maker_id=e.maker,
                      t=BASE_TIME+timedelta(seconds=e.counter-.5), content_ref=before_ref)
                e.add('Revision', revision_id=after_id, artifact_id=e.artifact, maker_id=e.maker,
                      parent_revision_ids=[before_id], t=BASE_TIME+timedelta(seconds=e.counter), content_ref=after_ref)
                e.rows['Artifact'][0]['revisions'].extend([before_id, after_id])
                e.add('Correction', correction_id=_identity(identity, 'correction', i), maker_id=e.maker,
                      decision_id=d['decision_id'], revision_id=after_id, before_ref=before_ref, after_ref=after_ref,
                      t=BASE_TIME+timedelta(seconds=e.counter))
            e.decision('choice', i, row['menu'], row['choice'])
        e.decision('query', 0, public['query'], None)
    else:
        raise NotImplementedError('family-specific canonical emitter not yet admitted: '+family)
    return e.finish()


def validate(bundle, schema, *, canaries=()):
    if bundle['schema_version'] != schema.SCHEMA_VERSION or bundle['schema_hash'] != schema_hash(schema):
        raise ValueError('schema mismatch')
    if bundle['transport'] != TRANSPORT:
        raise ValueError('transport mismatch')
    data = canonical(bundle)
    if any(token.encode() in data for token in canaries):
        raise ValueError('truth canary leaked')
    tables = {}
    for kind, values in bundle['rows'].items():
        if kind not in schema.RECORDS:
            raise ValueError('noncanonical record kind')
        tables[kind] = {}
        for value in values:
            row = schema.RECORDS[kind].model_validate(value)
            key = getattr(row, schema.ID_FIELDS[kind])
            if kind == 'Provenance':
                key = (row.record_type, key)
            if key in tables[kind]:
                raise ValueError('duplicate record')
            tables[kind][key] = row
    def require(kind, key):
        if key not in tables.get(kind, {}):
            raise ValueError('unresolved '+kind)
        return tables[kind][key]
    for kind, values in tables.items():
        if kind == 'Provenance':
            for record in values.values():
                require(record.record_type, record.record_id)
            continue
        for key, row in values.items():
            require('Provenance', (kind, key))
            if row.supersedes is not None:
                if row.supersedes == key: raise ValueError('self supersession')
                require(kind, row.supersedes)
            if hasattr(row, 'maker_id') and row.maker_id is not None:
                require('Maker', row.maker_id)
            if hasattr(row, 'artifact_id') and kind != 'Artifact':
                require('Artifact', row.artifact_id)
            if kind == 'Artifact':
                for m in row.maker_ids: require('Maker', m)
                for r in row.revisions:
                    if require('Revision', r).artifact_id != row.artifact_id:
                        raise ValueError('revision belongs to another artifact')
            if kind == 'Revision':
                for parent in row.parent_revision_ids:
                    previous = require('Revision', parent)
                    if previous.artifact_id != row.artifact_id or previous.t > row.t or parent == key:
                        raise ValueError('invalid revision parent')
                for goal in row.stated_goal_ids: require('StatedGoal', goal)
            if kind == 'Session':
                for goal in row.stated_goal_ids: require('StatedGoal', goal)
            if kind == 'StatedGoal':
                require({'session':'Session','artifact':'Artifact','revision':'Revision','decision':'Decision'}[row.scope], row.scope_id)
            if kind == 'Decision':
                session = require('Session', row.session_id)
                if row.maker_id != session.maker_id or not session.t_start <= row.t_open <= row.t_resolve <= session.t_end:
                    raise ValueError('invalid decision session')
                if row.outcome or set(row.context) != {'channel', 'index', 'observed'}:
                    raise ValueError('forbidden prediction fields')
                if row.context['channel'] == 'query' and row.taken is not None:
                    raise ValueError('query target leaked')
            if kind == 'ContextEvent':
                session = require('Session', row.session_id)
                if not session.t_start <= row.t <= session.t_end: raise ValueError('context outside session')
            if kind == 'Correction':
                if row.decision_id is not None:
                    decision = require('Decision', row.decision_id)
                    if decision.context['channel']=='query':raise ValueError('future query correction leaked')
                    if row.t < decision.t_resolve or row.maker_id != decision.maker_id:
                        raise ValueError('invalid correction chronology/attribution')
                if row.revision_id is not None:
                    revision = require('Revision', row.revision_id)
                    if revision.maker_id != row.maker_id or revision.t > row.t:
                        raise ValueError('invalid correction revision chronology/attribution')
            refs = [getattr(row, key, None) for key in ('content_ref', 'before_ref', 'after_ref')]
            refs.extend(o.content_ref for o in getattr(row, 'options', []))
            for ref in filter(None, refs):
                text = bundle['blobs'].get(ref)
                if text is None or hashlib.sha256(text.encode()).hexdigest() != ref:
                    raise ValueError('content reference mismatch')
    return tables


def decode(v, bundle, schema):
    validate(bundle, schema)
    decisions = bundle['rows']['Decision']; family = v['family']
    if family == 'N1': return decode_population(v, bundle, schema)
    by_channel = {}
    for row in decisions:
        c = row['context']; options = sorted(row['options'], key=lambda o: o['position'])
        taken = next((i for i, o in enumerate(options) if o['option_id'] == row['taken']), None)
        entry = (c['index'], [json.loads(bundle['blobs'][o['content_ref']]) for o in options], taken, c['observed'])
        by_channel.setdefault(c['channel'], []).append(entry)
    for rows in by_channel.values(): rows.sort(key=lambda x: x[0])
    if family == 'R1':
        return dict(own=by_channel['own'][0][2], claim=by_channel['claim'][0][2],
                    selection=[r[2] for r in by_channel['selection']], operations=[r[2] for r in by_channel['operations']])
    if family == 'R2':
        obs = [[0, 0, 0] for _ in range(v['settings']['prefix'])]
        for index, _, taken, _ in by_channel['observation']: obs[index//3][index%3] = taken
        return dict(observations=obs, flags=[r[2] for r in by_channel['flag']])
    if family == 'R3':
        def row(item, chosen):
            _, options, taken, observed = item
            if set(observed) != {'h'}: raise ValueError('unrecognized R3 context')
            out = dict(h=observed['h'], effects=[o['effect'] for o in options], costs=[o['cost'] for o in options])
            if chosen: out['picked'] = taken
            return out
        return dict(history=[row(item, True) for item in by_channel['history']], current=row(by_channel['query'][0], False))
    if family == 'R4':
        from .repairs import tree
        evidence = [-1]*len(tree(v['settings']['branching'])[0])
        for index, _, taken, _ in by_channel.get('leaf', []): evidence[index] = taken
        cue = by_channel['cue'][0]
        return dict(evidence=evidence, cue=cue[2], cue_node=cue[0])
    if family in ('N2', 'N5', 'N8'):
        result = dict(history=[dict(observed, choice=taken) for _, _, taken, observed in by_channel.get('history', [])])
        allowed = {'N2': {'latency_bin', 'context', 'class_cue', 'opponent', 'previous'},
                   'N5': {'quality'}, 'N8': {'threat', 'signal'}}[family]
        for row in result['history']:
            if set(row) != allowed | {'choice'}: raise ValueError('unrecognized history context')
        if family == 'N5':
            result['query'] = by_channel['query'][0][3]
            if set(result['query']) != {'quality'}: raise ValueError('unrecognized query context')
        return result
    if family == 'N6':
        return dict(history=[dict(claim=a[2], procedure=b[2]) for a, b in
            zip(by_channel['claim'], by_channel['procedure'], strict=True)])
    if family == 'N7':
        statements = sorted(bundle['rows']['StatedPreference'], key=lambda r: int(r['attribute'].split('-')[-1]))
        return dict(history=[dict(options=options, choice=taken) for _, options, taken, _ in by_channel.get('history', [])],
                    statement=[r['value'] for r in statements], query=by_channel['query'][0][1])
    if family == 'N4':
        if v['settings']['extension']:
            return dict(history=[[taken, observed['context']] for _, _, taken, observed in by_channel['history']],
                        changed=by_channel['query'][0][3]['changed'])
        return dict(history=[dict(options=options, choice=taken, **observed) for _, options, taken, observed in by_channel['history']],
                    query=dict(options=by_channel['query'][0][1], **by_channel['query'][0][3]))
    if family == 'N3':
        rows = []
        for edit, choice in zip(by_channel['edit'], by_channel['choice'], strict=True):
            _, options, picked, observed = edit
            rows.append(dict(x0=observed['x0'], context=observed['context'], picked=picked,
                             edits=[o['delta'] for o in options], costs=[o['cost'] for o in options],
                             menu=choice[1], choice=choice[2]))
        return dict(history=rows, query=by_channel['query'][0][1])
    raise NotImplementedError(family)


def emit_population(v, public, schema, identity):
    combined = dict(schema_version=schema.SCHEMA_VERSION, schema_hash=schema_hash(schema), transport=TRANSPORT,
                    rows={kind: [] for kind in schema.RECORDS}, blobs={})
    for index in range(len(public['choices'])):
        e = Emission(schema, [identity, 'population-member', index])
        e.rows['Maker'][0]['identity_basis'] = ['observed-population-index:'+str(index)]
        for i, (menu, chosen) in enumerate(zip(public['menus'][index], public['choices'][index], strict=True)):
            e.decision('history', i, np.asarray(menu).tolist(), int(chosen))
        alias = None if public['alias_pairs'] is None else np.asarray(public['alias_pairs'][index]).tolist()
        e.decision('query', 0, np.asarray(public['own'][index]).tolist(), None, observed={'context': 'own', 'alias_pair': alias})
        e.decision('query', 1, np.asarray(public['other'][index]).tolist(), None, observed={'context': 'other', 'alias_pair': alias})
        item = e.finish()
        for kind, rows in item['rows'].items(): combined['rows'][kind].extend(rows)
        combined['blobs'].update(item['blobs'])
    return combined


def decode_population(v, bundle, schema):
    makers = sorted(bundle['rows']['Maker'], key=lambda r: int(r['identity_basis'][0].split(':')[1]))
    menus, choices, own, other, aliases = [], [], [], [], []
    for maker in makers:
        rows = [r for r in bundle['rows']['Decision'] if r['maker_id'] == maker['maker_id']]
        history = sorted((r for r in rows if r['context']['channel'] == 'history'), key=lambda r: r['context']['index'])
        query = sorted((r for r in rows if r['context']['channel'] == 'query'), key=lambda r: r['context']['index'])
        def options(row):
            ordered = sorted(row['options'], key=lambda x: x['position'])
            return [json.loads(bundle['blobs'][o['content_ref']]) for o in ordered], next(
                (i for i, o in enumerate(ordered) if row['taken'] == o['option_id']), None)
        decoded = [options(r) for r in history]
        menus.append([r[0] for r in decoded]); choices.append([r[1] for r in decoded])
        own.append(options(query[0])[0]); other.append(options(query[1])[0]); aliases.append(query[0]['context']['observed']['alias_pair'])
    return dict(menus=np.asarray(menus), choices=np.asarray(choices), own=np.asarray(own), other=np.asarray(other),
                alias_pairs=None if all(a is None for a in aliases) else np.asarray(aliases))


def export(directory, bundle):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    for kind, rows in bundle['rows'].items():
        with (directory/(kind+'.jsonl')).open('xb') as stream:
            stream.write(canonical(dict(header=dict(transport=TRANSPORT, record_type=kind,
                schema_version=bundle['schema_version'], schema_hash=bundle['schema_hash'])))+b'\n')
            for row in rows: stream.write(canonical(row)+b'\n')
    write(directory/'SYNTHETIC_CONTENT.json', bundle['blobs'])


class ReferenceModule:
    """Engine-shaped interface. The evaluator truth is deliberately unavailable."""
    def __init__(self, variant, schema):
        self.variant, self.schema = deepcopy(variant), schema

    def fit(self, ledger, population_prior=None):
        if population_prior is not None:
            raise ValueError('this reference uses its declared finite prior')
        public=decode(self.variant, ledger, self.schema)
        if self.variant['family']=='N1':
            from .makers import n1_infer
            predictions,details=n1_infer(self.variant,public)
        else:
            from .dispatch import evaluate
            predictions,details=evaluate(self.variant,public)
        return dict(predictions=predictions,details=details,schema_hash=schema_hash(self.schema),
            makers=[row['maker_id'] for row in ledger['rows']['Maker']],
            queries={row['decision_id']:row for row in ledger['rows']['Decision'] if row['context']['channel']=='query'},
            population_indices={row['maker_id']:int(row['identity_basis'][0].split(':')[1])
                for row in ledger['rows']['Maker']} if self.variant['family']=='N1' else {})

    def predict(self, posterior, decision=None):
        if posterior['schema_hash']!=schema_hash(self.schema):raise ValueError('reference posterior schema mismatch')
        predictions = posterior['predictions']
        key = 'approximate-hierarchical' if self.variant['family'] == 'N5' else 'law-marginal'
        if decision is None:return predictions[key]
        row=decision.model_dump(mode='json') if hasattr(decision,'model_dump') else decision
        if posterior['queries'].get(row['decision_id'])!=row:raise ValueError('decision is outside fitted prediction boundary')
        if self.variant['family']=='N1':
            index=posterior['population_indices'][row['maker_id']]
            if row['context']['observed']['context']=='own':return posterior['details']['readers'][key]['in_context'][index]
            return predictions[key][index]
        return predictions[key][0]

    def explain(self, posterior, maker_id):
        if maker_id not in posterior['makers']:raise ValueError('unknown maker')
        return dict(scope='constructed generator; declared supplied priors, not human intent',
                    hypotheses=posterior['details'],maker_id=maker_id)
