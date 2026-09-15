"""Bounded read-only sample verification; no scientific stage or learner is invoked."""
import argparse,hashlib,json,math,sqlite3,sys,time,zipfile,zlib
from datetime import datetime,timezone
from pathlib import Path

parser=argparse.ArgumentParser()
for name in ['run-root','archive','manifest','scratch','out']:
 parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args()
root=args.run_root.resolve();source=args.scratch.resolve()
assert not source.exists() and root not in source.parents
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads(args.manifest.read_bytes())
assert sha(args.archive.read_bytes())==manifest['archive_sha256']
source.mkdir(parents=True)
with zipfile.ZipFile(args.archive) as archive:
 for name,expected in manifest['members'].items():
  target=(source/name).resolve()
  assert source in target.parents
  data=archive.read(name)
  assert sha(data)==expected==sha((root/'source'/name).read_bytes())
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
sys.path.insert(0,str(source))
from ghostscale.validation.soundingline.v17.continuation_analysis import verify_item,paired_value,unit_statistics
plan=json.loads((root/'PLAN.json').read_bytes())
window=json.loads((root/'WINDOW.json').read_bytes())
assert plan['source_files']==manifest['members']
assert sha((root/'CONTROLLER.json').read_bytes())==plan['controller_sha256']
units=[];snapshots=[]
with sqlite3.connect((root/'records.sqlite').as_uri()+'?mode=ro',uri=True,timeout=5) as db:
 db.execute('PRAGMA query_only=ON')
 until=time.monotonic()+30
 db.set_progress_handler(lambda:time.monotonic()>until,10000)
 db.execute('BEGIN')
 for key,expected in [('plan',plan),('window',window)]:
  assert json.loads(db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()[0])==expected
 for stage,designs in [('expansion',plan['initial_expansions']),('confirmation',plan['confirmation']),('robustness',plan['initial_expansions'])]:
  for design in designs:
   orders=['ASC','DESC'] if stage!='robustness' else ['DESC']
   for order in orders:
    row=db.execute('SELECT ci,digest,body,stats,paired FROM units WHERE stage=? AND branch=? ORDER BY ci '+order+' LIMIT 1',(stage,design['id'])).fetchone()
    assert row is not None,(stage,design['id'])
    units.append((stage,design,row))
 for name,body,expected in db.execute('SELECT name,body,digest FROM snapshots'):
  data=zlib.decompress(body)
  assert sha(data)==expected
  if (root/(name+'.json')).exists():assert json.loads(data)==json.loads((root/(name+'.json')).read_bytes())
  snapshots.append(dict(name=name,payload_sha256=expected))
 db.row_factory=sqlite3.Row
 failures=[dict(r) for r in db.execute('SELECT * FROM failures ORDER BY at,id')]
 totals=dict(db.execute('SELECT * FROM totals WHERE singleton=1').fetchone())
 db.rollback()
checks=[];cases=rows=0
for stage,design,(ci,expected,body,stats,paired) in units:
 data=zlib.decompress(body);assert sha(data)==expected
 items=json.loads(data)
 for item in items:verify_item(item)
 assert unit_statistics(items)==json.loads(zlib.decompress(stats))
 if stage=='confirmation':assert math.isclose(paired_value(items,design),paired,rel_tol=1e-12,abs_tol=1e-12)
 cases+=len(items);rows+=sum(len(i['rows']) for i in items)
 checks.append(dict(stage=stage,branch=design['id'],constructor_index=ci,unit_sha256=expected,cases=len(items),rows=sum(len(i['rows']) for i in items)))
assert all(sha((root/'source'/n).read_bytes())==h for n,h in manifest['members'].items())
assert window==json.loads((root/'WINDOW.json').read_bytes())
report=dict(schema='v17.bounded-validity-audit.1',observed_at=datetime.now(timezone.utc).isoformat(),passed=True,source_members_verified=len(manifest['members']),source_archive_sha256=manifest['archive_sha256'],unit_selection='First and last retained unit per expansion/confirmation branch; latest retained unit per robustness branch, including the halted branch.',sample_units=len(checks),sample_cases=cases,sample_rows=rows,checks=checks,snapshots=snapshots,failures=failures,execution_totals_at_snapshot=totals,immutable_window=window,live_source_unchanged=True,live_database_access='read-only; query_only; bounded transaction; indexed unit lookups',scientific_stage_launched=False,scope='Rechecked stored unit and snapshot hashes, independent projection/physics/probability/proper-score/cost rules, stored unit summaries and sampled paired confirmation values.',limitations='Deterministic spot check, not a random severity estimate or complete independent reaggregation; no final scientific claim. Retained branch failure remains unresolved.',audit_script_sha256=sha(Path(__file__).read_bytes()))
args.out.parent.mkdir(parents=True,exist_ok=True)
with args.out.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:report[k] for k in ['passed','sample_units','sample_cases','sample_rows','source_members_verified','failures','limitations']}))
