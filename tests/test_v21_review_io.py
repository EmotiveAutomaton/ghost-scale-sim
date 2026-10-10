import numpy as np
from ghostscale.validation.soundingline.v18_3.io import write,file_digest
from ghostscale.validation.soundingline.v21.metrics import scores
from runners import review_v21

def test_archive_arrays_loaded_once_before_scalar_check(tmp_path,monkeypatch):
    p=np.full((3,128),1/128);lp=np.log(p);y=np.array([1,2,3]);a=scores(lp,y)
    raw=tmp_path/'raw';raw.mkdir();target=raw/'fixture_points.npz'
    np.savez_compressed(target,logp=lp,truth=y,**a)
    write(tmp_path/'PLAN.json',{'fixture':True})
    write(tmp_path/'COMPLETE.json',dict(plan_sha256=file_digest(tmp_path/'PLAN.json'),files={'raw/fixture_points.npz':file_digest(target)}))
    original=np.load;reads={}
    class Counted:
        def __init__(self,path):self.archive=original(path);self.files=self.archive.files
        def __enter__(self):return self
        def __exit__(self,*args):self.archive.close()
        def __contains__(self,key):return key in self.files
        def __getitem__(self,key):reads[key]=reads.get(key,0)+1;return self.archive[key]
    monkeypatch.setattr(review_v21.np,'load',Counted)
    assert review_v21.verify(tmp_path)['forecast_rows']==3
    assert reads and max(reads.values())==1
