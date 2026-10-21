import ast
from pathlib import Path
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]

class IntegrityTests(unittest.TestCase):
    def test_python_syntax(self):
        for p in ROOT.rglob('*.py'):
            if any(x in p.parts for x in ['_artifacts','__pycache__','.venv']): continue
            with self.subTest(path=str(p.relative_to(ROOT))): ast.parse(p.read_text(encoding='utf-8-sig'))

    def test_reference_configs(self):
        configs=list((ROOT/'HS-AE').glob('configs/*.yaml'))+list((ROOT/'HSI_Blood_mc').glob('configs/*.yaml'))
        for p in configs:
            c=yaml.safe_load(p.read_text(encoding='utf-8'))
            with self.subTest(config=p.name):
                self.assertEqual(c['model']['cfg']['hidden_dims'],[256,128])
                self.assertEqual(c['model']['cfg']['latent_dim'],64)
                self.assertEqual(c['logging']['monitor'],'val/loss')
                self.assertEqual(c['logging']['mode'],'min')
                self.assertEqual(c['data']['preprocess']['dwt']['wavelet'],'db4')
                self.assertNotIn('/home/',c['data']['csv_path'])
                self.assertAlmostEqual(sum(c['data'][x] for x in ['train_pct','val_pct','test_pct']),1.)

if __name__=='__main__': unittest.main()
