"""V31 seeds, immutable records and independent whole-world summaries."""
import json,zlib
import numpy as np
from ..v30.common import run_cell,rows_for,save_npz
from ..v27.common import read,write,digest
from ..v29.common import mean_blocks,censored_blocks,sustained

def rng(*parts):
    return np.random.default_rng(zlib.crc32(json.dumps(['V31',*parts],sort_keys=True,separators=(',',':')).encode()))
