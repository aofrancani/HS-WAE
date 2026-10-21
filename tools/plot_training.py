"""Plot TensorBoard training/validation loss from a selected run directory."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logdir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Output PNG or PDF path')
    p.add_argument('--tags',nargs='+',default=['train/loss','val/loss'])
    a=p.parse_args(); events=EventAccumulator(str(a.logdir)); events.Reload()
    available=events.Tags()['scalars']; fig,ax=plt.subplots(figsize=(6,4))
    for tag in a.tags:
        if tag not in available: p.error(f'{tag} absent. Available scalar tags: {available}')
        values=events.Scalars(tag); ax.plot([v.step for v in values],[v.value for v in values],marker='o' if len(values)<3 else None,label=tag)
    ax.set(xlabel='Global training step',ylabel='Loss'); ax.legend(); fig.tight_layout()
    a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=180); plt.close(fig)

if __name__=='__main__': main()
