"""Point the unchanged upstream loader at a downloaded HyperBlood directory."""
import argparse
import importlib
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',required=True,help='Directory containing data/ and anno/.')
    parser.add_argument('--task',choices=['extract','figure'],default='extract')
    args,remaining=parser.parse_known_args()
    root=Path(args.dataset_root).resolve()
    for sub in ['data','anno']:
        if not (root/sub).is_dir(): parser.error(f'Missing {root/sub}')
    module=importlib.import_module('preprocessing.hyperblood.'+('data_preparation' if args.task=='extract' else 'generate_fig_paper'))
    sys.modules['ds_load'].PATH_DATA=root.as_posix()+'/'
    sys.argv=[module.__file__,*remaining]
    module.main()

if __name__=='__main__': main()
