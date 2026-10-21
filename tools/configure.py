"""Create local config copies without modifying reference configurations."""
import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soybean-root', type=Path)
    parser.add_argument('--blood-csv', type=Path)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing generated local configs.')
    args = parser.parse_args(argv)
    if args.num_workers is not None and args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if args.soybean_root is not None:
        if not (args.soybean_root / 'soybean_HSI.csv').is_file():
            parser.error('--soybean-root must contain soybean_HSI.csv')
    if args.blood_csv is not None and not args.blood_csv.is_file():
        parser.error('--blood-csv must point to an existing CSV file')

    # Prepare and validate the entire batch before creating any local copies.
    pending = []
    for project in ['HS-AE', 'HSI_Blood_mc']:
        working = ROOT / project
        for folder in [working / 'configs', working / 'v0/configs']:
            for source in sorted(folder.glob('*.yaml')):
                config = yaml.safe_load(source.read_text(encoding='utf-8'))
                data = args.soybean_root if project == 'HS-AE' else args.blood_csv
                source_cwd = (
                    working / 'v0'
                    if project == 'HS-AE' and folder.parent.name == 'v0'
                    else working
                )
                existing = (source_cwd / config['data']['csv_path']).resolve()
                config['data']['csv_path'] = str(data.resolve() if data else existing)
                if args.cpu:
                    config['trainer'].update(accelerator='cpu', devices='auto')
                if args.num_workers is not None:
                    config['data']['num_workers'] = args.num_workers
                destination = folder / 'local' / source.name
                if destination.exists() and not args.overwrite:
                    parser.error(f'{destination} exists; use --overwrite to replace it')
                pending.append((destination, yaml.safe_dump(config, sort_keys=False)))

    for destination, content in pending:
        destination.parent.mkdir(exist_ok=True)
        with destination.open('w' if args.overwrite else 'x', encoding='utf-8') as stream:
            stream.write(content)
        print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
