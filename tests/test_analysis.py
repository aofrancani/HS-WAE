import unittest
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

def source_helpers():
    path=Path(__file__).resolve().parents[1]/'HS-AE/scripts/finalize_wtam_definitive.py'
    spec=importlib.util.spec_from_file_location('titan_statistics',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class AnalysisTests(unittest.TestCase):
    def test_insufficient_finite_pairs_rejected(self):
        h=source_helpers()
        with self.assertRaises(RuntimeError): h.paired_stats([1,float('nan')],'incomplete')

    def test_pairing_averages_days_within_seed(self):
        rows=[]
        for seed in [2,3,4]:
            for day in [1,2]:
                rows.extend([{'variant':'a','seed':seed,'day':day,'f1':.1}, {'variant':'b','seed':seed,'day':day,'f1':.1+seed*.01+day*.001}])
        frame=pd.DataFrame(rows).sample(frac=1,random_state=2)
        h=source_helpers()
        values=h.architecture_benefit(frame,'a','b','f1',units=('day',))
        np.testing.assert_allclose(values,[.0215,.0315,.0415])
        self.assertEqual(h.paired_stats(values,'fixture')['n_pairs'],3)

    def test_holm_monotonic_and_bounded(self):
        h=source_helpers(); rows=[{'p_raw':.01},{'p_raw':.04},{'p_raw':.03}]; h.holm(rows)
        np.testing.assert_allclose([r['p_holm'] for r in rows],[.03,.06,.06])
        self.assertEqual(h.paired_stats([0,0,0],'zero')['p_raw'],1)

if __name__=='__main__': unittest.main()
