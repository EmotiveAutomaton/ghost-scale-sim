"""Idempotent, local full write-through of separately named exploratory maps."""
import json
import math
from pathlib import Path
from .common import read,write,digest
from . import registry


def hypothesis_note(family):
    return (f'Friday exploratory maps for {family} remain separate descriptive '
            'evidence; their wider conditions do not promote the original result.')


def place_hypothesis_note(lines, family, anchor):
    """Keep the qualification separate from a possibly multi-line synthesis paragraph."""
    note=hypothesis_note(family)
    for i,line in enumerate(lines):
        if note in line:
            lines[i]=line.replace(note,'').rstrip()
    end=next((i for i in range(anchor+2,len(lines)) if lines[i].startswith('**What these add up to.**')),None)
    if end is not None:
        lines[end:end]=[note,'']


def repair_hypothesis_notes(path):
    """Repair output from the earlier writer without changing rows or synthesis prose."""
    lines=path.read_text(encoding='utf-8').splitlines()
    for family in ('E1','H1','H2','H3','R1','U1','E2','E3'):
        anchor=next((i for i,line in enumerate(lines) if line.startswith('| G-V26-'+family+'-1 |')),None)
        if anchor is not None and any(hypothesis_note(family) in line for line in lines):
            place_hypothesis_note(lines,family,anchor)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def featured(summary):
    f=summary['family'];out=[]
    for c in summary['cells']:
        l=c['labels'];m=c['metric'];keep=False
        if f=='H1':keep=l==dict(reader='contrasts',history=4) and m=='primary'
        elif f in ('H2','H3'):keep=l.get('reader')=='contrasts' and m=='primary'
        elif f=='E1':keep=l.get('reader')=='B-over-A-persistence' and m in ('ever_crossed','final_above','recrossed_below') and l['history']==summary['settings'][c['setting']]['max_history']
        elif f=='E2':keep=l.get('reader')=='contrasts' and l.get('history')==8 and m=='primary'
        elif f=='E3':keep=l.get('reader')=='contrasts' and l.get('history')==32 and m=='primary'
        elif f=='U1':keep=l.get('truth')==0 and l.get('history') in (16,24,32,64) and m=='excess_capture'
        elif f=='R1':keep=l.get('reader')=='switch' and l.get('history') in (1,16) and m=='primary'
        if keep:out.append(c)
    return out


def write_through(repo, claim, wording, relative, text, estimate, family):
    """One controller is the writer; partial landing can be resumed without duplicates."""
    path=repo/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('existing report differs; explicit correction required')
    path.write_text(text,encoding='utf-8',newline='\n')
    exchange=repo/'docs/exchange'/('GHOST_'+claim.replace('-','_')+'.md')
    exchange.write_text('# '+claim+'\n\n'+wording+'\n\n'+ '\n\n'.join(registry.ROUTES[family])+'\n\n[Full report](../'+relative.removeprefix('docs/')+').\n',encoding='utf-8',newline='\n')
    link=relative if relative.startswith('docs/') else relative
    p=repo/'FINDINGS.md';content=p.read_text(encoding='utf-8');mark='## '+claim
    if mark not in content:
        i=content.find('\n');content=content[:i+1]+f'\n{mark}\n\n{wording}\n\n[Report]({link}).\n'+content[i+1:];p.write_text(content,encoding='utf-8',newline='\n')
    p=repo/'docs/theory/READING_INTENT.md';content=p.read_text(encoding='utf-8')
    if '| '+claim+' |' not in content:
        lines=content.splitlines();anchor=next((i for i,line in enumerate(lines) if line.startswith('| G-V26-'+family+'-1 |')),None)
        entry=f'| {claim} | Separate Friday exploratory map | **QUALIFIED descriptive (run, constructed method)** | {wording} [Report](../{relative.removeprefix("docs/")}) |'
        if anchor is None:raise ValueError('parent hypothesis row missing: '+family)
        lines.insert(anchor+1,entry)
        place_hypothesis_note(lines,family,anchor)
        p.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    claims=read(repo/'docs/theory/CLAIMS.json');existing=next((x for x in claims['claims'] if x['id']==claim),None)
    row=dict(id=claim,allowed_wording=wording,claim_kind='method',architecture='miniature, architecture untested',assumed_choice_model=registry.CHOICES[family],reader='Declared finite-library readers; see separate protocol',estimate=estimate,control_status='passes',source_file=relative,source_line=3,source_sha256=digest(path),supersedes=None,superseded_by=None,correction_class=None)
    if existing and existing!=row:raise ValueError('existing claim differs')
    if not existing:claims['claims'].append(row);write(repo/'docs/theory/CLAIMS.json',claims,immutable=False)


def land(campaign,packet,repo):
    if (packet/'LOCAL_LANDING.json').exists():return
    from runners.run_v26_followups import verify_complete
    verify_complete(packet);plan=read(packet/'PLAN.json')
    if plan['scientific_kind']=='probe':return
    summary=read(packet/'original/SUMMARY.json');protocol=read(campaign/'protocols'/(plan['card']+'.json'));cells=featured(summary)
    family=plan['family'];claim='G-V26-FR-'+plan['card'].replace('-','')+'-1'
    relative='docs/versions/v26-top-and-ends/friday-20261104/'+plan['card']+'_REPORT.md'
    groups={}
    for c in cells:
        if c['mean'] is not None:groups.setdefault(c['metric'],[]).append(c['mean'])
    units='probabilities' if family in ('E1','U1') else 'nats per deliberation' if family=='H2' else 'nats per revision' if family=='H3' else 'nats'
    ranges='; '.join(f'{k.replace("_"," ")} {min(v):.6g} to {max(v):.6g}' for k,v in groups.items()) or 'featured estimates unavailable'
    wording=f"{protocol['question']} The separate exploratory map retains {len(summary['settings'])} settings and {plan['blocks']} independent worlds per setting. Featured cell means ({units}) span: {ranges}. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested."
    lines=['# '+plan['card']+' exploratory map','',wording,'','Claim `'+claim+'`. Earlier finite results and closeouts remain unchanged.','',
        'The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.','',
        '| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |','|---|---|---|---:|---:|---:|']
    for c in cells:lines.append('| '+str(c['setting'])+' | '+json.dumps(c['labels'],sort_keys=True)+' | '+c['metric']+' | '+str(c['mean'])+' | '+str(c['standard_error'])+' | '+str(c['available_blocks'])+' |')
    lines+=['','Settings and prospective case requirement:','', '```json',json.dumps(dict(settings=summary['settings'],case=protocol['case']),indent=2),'```','',
        'Validity: all original/replay raw bytes match; scalar posterior reconstruction and separate decimal whole-world aggregation pass. The declared generative model and some likelihood kernels are shared. Known answers validate equations, not real-world mechanisms. Independent settings are not paired causal contrasts. No simultaneous uncertainty guarantee or confirmation is asserted.',
        '', 'Construction and exclusions: '+protocol['construction'], '',*protocol['routes'],'',
        f'[Complete aggregate](../../../../results/v26/friday-20261104/{packet.name}/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/{packet.name}/INDEPENDENT_REVIEW.json).','']
    write_through(repo,claim,wording,relative,'\n'.join(lines),cells,family)
    write(packet/'LOCAL_LANDING.json',dict(claim=claim,report=relative,report_sha256=digest(repo/relative),published=False,recipient_copy_pending=True))
