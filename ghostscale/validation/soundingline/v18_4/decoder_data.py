"""Fresh supervised behavioral readouts of an immutable learned encoder."""
from pathlib import Path
import shutil
from ..v18_3.io import read,write,file_digest
from . import neural_data as D


def prepare(root,parents,data,pulse=lambda **kw:None):
    root=Path(root);public=root/'reader'
    manifest=D.prepare(root,pulse=pulse,namespace='v18.4-decoder',**data)
    if manifest.get('encoders'):return manifest
    encoders={};parent_hashes={}
    for prefix,parent in parents.items():
        proof=read(parent/'INDEPENDENT_REPLAY.json')
        if not proof['passed'] or proof['summary_sha256']!=file_digest(parent/'SUMMARY.json'):
            raise ValueError('decoder requires verified parent')
        parent_hashes[prefix]=file_digest(parent/'COMPLETE.json')
        completed=read(parent/'neural/COMPLETE.json')
        for name,predictions in completed['predictions'].items():
            selected={p['selected'] for p in predictions.values()}
            if len(selected)!=1:raise ValueError('test-dependent encoder selection')
            fit=selected.pop();source=parent/'neural'/fit/'BEST.pt'
            if file_digest(source)!=completed['fits'][fit]['best_sha256']:raise ValueError('parent weights changed')
            name=prefix+'-'+name;target=public/f'{name}-ENCODER.pt';shutil.copyfile(source,target)
            encoders[name]=dict(name=target.name,sha256=file_digest(target))
    manifest.update(encoders=encoders,query_context_features=len(D.context_features(D.TRAIN_QUERIES[0])),
        parent_complete_sha256=parent_hashes,
        training_supervision='fresh equal diverse behavioral labels, with frozen old-question encoders; farther question family withheld',
        decoder_scope='recoverability with additional labels; no claim that original decoder used recovered information')
    write(public/'INPUTS.json',manifest,immutable=False)
    return manifest
