"""V30 independent seeds and per-cell retained evidence."""
import json
import zlib
from pathlib import Path
import numpy as np
from ..v29.common import mean_blocks, censored_blocks, conditional_blocks, sustained, audit_controls
from ..v27.common import read, write, digest

def rng(*parts):
    return np.random.default_rng(zlib.crc32(json.dumps(["V30", *parts],sort_keys=True,separators=(",",":")).encode()))

def save_npz(path, arrays):
    np.savez_compressed(path, **arrays)

def rows_for(s, measures):
    rows=[]
    for key,value in measures.items():
        if key.startswith("_"): continue
        a=np.asarray(value)
        if a.ndim != 1: continue
        if key.startswith("separation"):
            value=censored_blocks(a,s["steps"],s["blocks"])
        else: value=mean_blocks(a,s["blocks"])
        rows.append(dict(measure=key,**value))
    return rows

def run_cell(module, s, dest, mode="original", original=None, lane="study", pulse=lambda **kw:None):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if mode=="reconstruct":
        with np.load(Path(original)/"cell_points.npz",allow_pickle=False) as z:
            raw={k:z[k] for k in z.files}
    else:
        raw=module.generate(s,lane,pulse=pulse)
        save_npz(dest/"cell_points.npz",raw)
    measures=module.infer(raw,s,reference=mode=="reconstruct",pulse=pulse)
    save_npz(dest/"cell_measures.npz",measures)
    result=dict(setting=s,rows=module.summarize(s,measures),no_registered_claim=True,
                miniature="architecture untested")
    write(dest/"SUMMARY.json",result)
    if mode!="reconstruct": write(dest/"RAW_MANIFEST.json",{"cell_points.npz":digest(dest/"cell_points.npz")})
    return result
