"""Plot saved definitive-run arrays without loading a model or requiring LaTeX."""
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True,help='Run directories containing test_outputs.npz')
    p.add_argument('--labels',nargs='+')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--perplexity',type=float,default=30)
    a=p.parse_args()
    if a.labels and len(a.labels)!=len(a.runs): p.error('Supply one label per run.')
    a.outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,len(a.runs),figsize=(4*len(a.runs),3.8),squeeze=False)
    fig_rec,rec_axes=plt.subplots(len(a.runs),1,figsize=(7,3*len(a.runs)),squeeze=False)
    for i,run in enumerate(a.runs):
        z=np.load(run/'test_outputs.npz'); labels=a.labels or [r.parent.parent.name for r in a.runs]
        coords=TSNE(n_components=2,random_state=a.seed,perplexity=min(a.perplexity,len(z['latent'])-1),init='pca',learning_rate='auto').fit_transform(z['latent'])
        ax=axes[0,i]
        for cls in np.unique(z['y_true']):
            take=z['y_true']==cls; ax.scatter(*coords[take].T,s=14,label=str(cls))
        ax.set(title=labels[i],xlabel='t-SNE 1',ylabel='t-SNE 2'); ax.legend(title='Class',fontsize=7)
        rax=rec_axes[i,0]
        rax.plot(z['raw_target'][0],label='Input'); rax.plot(z['reconstruction'][0],label='Reconstruction')
        rax.set(title=labels[i]+' — first test sample',xlabel='Retained band index',ylabel='Reflectance'); rax.legend()
        np.savez_compressed(a.outdir/f'tsne_{i}.npz',coordinates=coords,y_true=z['y_true'])
    for f,name in [(fig,'tsne'),(fig_rec,'reconstruction')]:
        f.tight_layout(); f.savefig(a.outdir/(name+'.png'),dpi=180); f.savefig(a.outdir/(name+'.pdf')); plt.close(f)
    (a.outdir/'provenance.json').write_text(json.dumps({'runs':[str(r.resolve()) for r in a.runs],'seed':a.seed,'perplexity_requested':a.perplexity,'interpretation':'Qualitative only; quantitative latent metrics use original 64-dimensional embeddings.'},indent=2)+'\n')

if __name__=='__main__': main()
