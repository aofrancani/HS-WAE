"""Subprocess checks with persistent reports, including failures to start."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def save_report(path, report):
    """Replace a report atomically so readers never see partially written JSON."""
    path = Path(path)
    checks = report['checks']
    report['passed'] = sum(item['exit_code'] == 0 for item in checks)
    report['failed'] = sum(item['exit_code'] != 0 for item in checks)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_check(name, cwd, args, *, env, outdir, report_path, report, timeout=600):
    """Record success, nonzero exit, timeout, or an executable launch error."""
    command = [str(argument) for argument in args]
    log = Path(outdir) / (name + '.log')
    start = time.monotonic()
    item = {'name': name, 'command': command, 'cwd': str(cwd),
            'exit_code': None, 'log': str(log)}
    with log.open('w', encoding='utf-8') as stream:
        try:
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
            item['exit_code'] = result.returncode
            item['status'] = 'passed' if result.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            item.update(status='timeout', error=f'Exceeded {timeout} seconds')
        except OSError as error:
            item.update(status='launch_error', error=str(error))
        if 'error' in item:
            stream.write('\n' + item['error'] + '\n')
    item['seconds'] = round(time.monotonic() - start, 2)
    report['checks'].append(item)
    save_report(report_path, report)
    passed = item['exit_code'] == 0
    print(('PASS ' if passed else 'FAIL ') + name, flush=True)
    return passed
