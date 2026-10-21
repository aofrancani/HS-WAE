"""Regression checks for the new execution and configuration adapters."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tests.command_runner import run_check
from tools import configure


ROOT = Path(__file__).resolve().parents[1]


class CommandRunnerTests(unittest.TestCase):
    def run_fixture(self, arguments, timeout=5):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = {'checks': [], 'completed': False}
            env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
            with redirect_stdout(io.StringIO()):
                passed = run_check(
                    'fixture', directory, arguments, env=env, outdir=directory,
                    report_path=directory / 'report.json', report=report, timeout=timeout,
                )
            saved = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, report)
            self.assertFalse(saved['completed'])
            self.assertEqual(len(saved['checks']), 1)
            log = (directory / 'fixture.log').read_text(encoding='utf-8')
            return passed, saved, log

    def test_success_keeps_utf8_log(self):
        passed, report, log = self.run_fixture([sys.executable, '-c', "print('Françani')"])
        self.assertTrue(passed)
        self.assertEqual((report['passed'], report['failed']), (1, 0))
        self.assertIn('Françani', log)

    def test_nonzero_exit_is_failure(self):
        passed, report, _ = self.run_fixture([sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertFalse(passed)
        self.assertEqual(report['checks'][0]['exit_code'], 7)
        self.assertEqual(report['failed'], 1)

    def test_timeout_is_recorded(self):
        passed, report, log = self.run_fixture(
            [sys.executable, '-c', 'import time; time.sleep(60)'], timeout=0.1,
        )
        self.assertFalse(passed)
        self.assertEqual(report['checks'][0]['status'], 'timeout')
        self.assertEqual(report['failed'], 1)
        self.assertIn('Exceeded', log)

    def test_missing_executable_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            passed, report, _ = self.run_fixture([str(Path(temporary) / 'missing-python')])
        self.assertFalse(passed)
        self.assertEqual(report['checks'][0]['status'], 'launch_error')
        self.assertEqual(report['failed'], 1)


class ConfigureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.references = {}
        for project in ['HS-AE', 'HSI_Blood_mc']:
            for subfolder in ['configs', 'v0/configs']:
                folder = ROOT / project / subfolder
                for source in folder.glob('*.yaml'):
                    relative = source.relative_to(ROOT)
                    destination = self.root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    self.references[relative] = source.read_bytes()
        self.addCleanup(patch.stopall)
        patch.object(configure, 'ROOT', self.root).start()

    def invoke(self, *args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            configure.main(list(args))

    def test_local_paths_and_reference_bytes_are_preserved(self):
        self.invoke('--cpu', '--num-workers', '0')
        for relative, original_bytes in self.references.items():
            source = self.root / relative
            self.assertEqual(source.read_bytes(), original_bytes)
            config = yaml.safe_load((source.parent / 'local' / source.name).read_text())
            data = 'data/HS-Class' if relative.parts[0] == 'HS-AE' else 'data/HSI_Blood/HSI_Blood.csv'
            self.assertEqual(Path(config['data']['csv_path']), self.root / data)
            self.assertEqual(config['trainer']['accelerator'], 'cpu')
            self.assertEqual(config['data']['num_workers'], 0)

    def test_invalid_inputs_create_no_configs(self):
        for arguments in [('--num-workers', '-1'), ('--blood-csv', str(self.root / 'missing.csv')),
                          ('--soybean-root', str(self.root / 'missing'))]:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as error:
                self.invoke(*arguments)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(list(self.root.rglob('local')))

    def test_existing_local_settings_require_explicit_overwrite(self):
        target = self.root / 'HSI_Blood_mc/configs/local/wtam_definitive.yaml'
        target.parent.mkdir(parents=True)
        target.write_text('user settings\n', encoding='utf-8')
        with self.assertRaises(SystemExit) as error:
            self.invoke('--cpu')
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(target.read_text(), 'user settings\n')
        self.assertFalse((self.root / 'HS-AE/configs/local').exists())
        self.invoke('--cpu', '--overwrite')
        self.assertEqual(yaml.safe_load(target.read_text())['trainer']['accelerator'], 'cpu')


if __name__ == '__main__':
    unittest.main()
