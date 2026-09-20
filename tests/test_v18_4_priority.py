import os
import subprocess
import sys
import pytest


@pytest.mark.skipif(os.name!='nt',reason='Windows native priority fixture')
def test_native_priority_is_actually_applied():
    result=subprocess.run([sys.executable,'-B','-m','ghostscale.validation.soundingline.v18_4.priority'],
        capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.stdout.strip()=='16384'
