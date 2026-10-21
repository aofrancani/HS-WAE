#!/usr/bin/env python3
"""
DWT-level energy vs discriminability analysis.

Compare wavelet-level coefficient energy with class discriminability.

For each NIR acquisition day (or any requested band/day), this script:
1) rebuilds the fixed test split with DWT enabled;
2) recomputes RAW DWT coefficients from the unstandardized retained spectrum;
3) measures per-sample level energy (sum of squares) and RMS amplitude;
4) tests class differences in level energy using Mann-Whitney U + rank-biserial effect size;
5) independently measures multivariate class separability of each STANDARDIZED
   DWT level vector using Silhouette, Davies-Bouldin, centroid distance and a
   normalized separation ratio;
6) applies Holm correction across DWT levels within each acquisition day.

No model training is performed and no t-SNE projection is used.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import silhouette_score, davies_bouldin_score

from utils import load_config, override_band_region, override_date_filters
from datasets.build_dataloader import build_datamodule
from datasets.crop_dataset import _dwt_row_levels


def holm(p):
    p=np.asarray(p,float); out=np.full_like(p,np.nan)
    valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid])]
    m=len(order); prev=0.0
    for rank,idx in enumerate(order):
        v=min(1.0,(m-rank)*p[idx]); prev=max(prev,v); out[idx]=prev
    return out


def rank_biserial_mwu(u, n0, n1):
    # Positive means class 1 tends to have higher values than class 0.
    # scipy U returned below is for class0 vs class1, so orient accordingly.
    return float(1.0 - 2.0*u/(n0*n1)) if n0*n1 else np.nan


def sep_metrics(X,y):
    cls=np.unique(y)
    if len(cls)!=2: return {k:np.nan for k in ['silhouette','db','centroid_distance','separation_ratio']}
    try: sil=float(silhouette_score(X,y))
    except Exception: sil=np.nan
    try: db=float(davies_bouldin_score(X,y))
    except Exception: db=np.nan
    X0=X[y==cls[0]]; X1=X[y==cls[1]]
    mu0=X0.mean(0); mu1=X1.mean(0)
    cd=float(np.linalg.norm(mu1-mu0))
    r0=float(np.sqrt(np.mean(np.sum((X0-mu0)**2,axis=1))))
    r1=float(np.sqrt(np.mean(np.sum((X1-mu1)**2,axis=1))))
    sr=cd/max((r0+r1)/2.0,1e-12)
    return {'silhouette':sil,'db':db,'centroid_distance':cd,'separation_ratio':sr}


def main(args):
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; sample_rows=[]
    for day in args.days:
        cfg=copy.deepcopy(load_config(Path(args.config)))
        if args.data_root: cfg['data']['csv_path']=args.data_root
        cfg.setdefault('data',{}).setdefault('preprocess',{}).setdefault('dwt',{})['enabled']=True
        override_band_region(cfg,args.band_region); override_date_filters(cfg,[day])
        dm=build_datamodule(cfg)
        ds=dm.test_ds
        y=ds.y.detach().cpu().numpy().astype(int)
        raw=ds.X_raw.detach().cpu().numpy()
        level_names=list(dm.feature_names)
        # standardized level vectors are already stored by DataModule
        std_levels=[[np.asarray(v,dtype=np.float32) for v in sample] for sample in ds.X_levels]

        # Raw DWT coefficient vectors for physical energy analysis.
        raw_levels=[]
        for x in raw:
            raw_levels.append(_dwt_row_levels(
                x,
                wavelet=dm.cfg.dwt_wavelet,
                level=dm.cfg.dwt_level,
                boundary=dm.cfg.dwt_boundary,
                include_approx=dm.cfg.dwt_include_approx,
            ))

        day_rows=[]
        for li,name in enumerate(level_names):
            R=np.stack([r[li] for r in raw_levels])
            Z=np.stack([r[li] for r in std_levels])
            energy=np.sum(R**2,axis=1)
            rms=np.sqrt(np.mean(R**2,axis=1))
            e0=energy[y==0]; e1=energy[y==1]
            try:
                u,p=mannwhitneyu(e0,e1,alternative='two-sided')
                rb=rank_biserial_mwu(u,len(e0),len(e1))
            except Exception:
                u=p=rb=np.nan
            sm=sep_metrics(Z,y)
            row={
                'band_region':args.band_region,'day':day,'level_index':li,'level':name,
                'n_coeff':R.shape[1],
                'energy_mean_control':float(np.mean(e0)),'energy_mean_damaged':float(np.mean(e1)),
                'energy_median_control':float(np.median(e0)),'energy_median_damaged':float(np.median(e1)),
                'energy_u':float(u),'energy_p_raw':float(p),'energy_rank_biserial_damaged_minus_control':float(rb),
                'rms_mean_control':float(np.mean(rms[y==0])),'rms_mean_damaged':float(np.mean(rms[y==1])),
                **sm,
            }
            day_rows.append(row)
            for i in range(len(y)):
                sample_rows.append({'band_region':args.band_region,'day':day,'sample_index':i,'label':int(y[i]),
                    'level_index':li,'level':name,'energy':float(energy[i]),'rms':float(rms[i])})
        pp=np.array([r['energy_p_raw'] for r in day_rows],float)
        adj=holm(pp)
        for r,a in zip(day_rows,adj): r['energy_p_holm']=a
        all_rows.extend(day_rows)

    stats=pd.DataFrame(all_rows)
    samples=pd.DataFrame(sample_rows)
    stats.to_csv(out/'dwt_level_discriminability.csv',index=False)
    samples.to_csv(out/'dwt_level_sample_energy.csv',index=False)

    # Day-level correlation between average level energy and discriminability.
    corr=[]
    for day,g in stats.groupby('day'):
        energy=(g.energy_mean_control+g.energy_mean_damaged)/2
        for metric in ['silhouette','separation_ratio']:
            rho,p=spearmanr(energy,g[metric],nan_policy='omit')
            corr.append({'day':day,'metric':metric,'spearman_energy_vs_discriminability':rho,'p':p})
    pd.DataFrame(corr).to_csv(out/'energy_vs_discriminability_correlation.csv',index=False)

    # Compact figures. Energy is normalized within each day only for visual comparison.
    figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    for day,g in stats.groupby('day'):
        g=g.sort_values('level_index')
        avg_e=((g.energy_mean_control+g.energy_mean_damaged)/2).to_numpy(float)
        norm_e=avg_e/max(np.nanmax(avg_e),1e-12)
        x=np.arange(len(g))
        plt.figure(figsize=(7,4.2))
        plt.plot(x,norm_e,marker='o',label='Normalized mean energy')
        plt.plot(x,g.silhouette.to_numpy(float),marker='s',label='Silhouette (level vector)')
        plt.xticks(x,g.level,rotation=45,ha='right'); plt.xlabel('DWT level'); plt.ylabel('Value')
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(figdir/f'{args.band_region}_day_{day:02d}_energy_vs_silhouette.pdf',bbox_inches='tight')
        plt.close()

    report=[
        '# DWT level energy vs discriminability', '',
        'Energy and discriminability are evaluated separately. Raw coefficient energy is not used as a proxy for class relevance.',
        'Class-energy differences use Mann-Whitney U tests with Holm correction across DWT levels within each day.',
        'Multivariate level separability is calculated directly in each standardized DWT coefficient vector; no t-SNE is used.',
    ]
    (out/'dwt_level_report.md').write_text('\n'.join(report),encoding='utf-8')
    print('Saved',out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/dwt_attn.yaml')
    p.add_argument('--data-root',default=None)
    p.add_argument('--band_region',default='nir')
    p.add_argument('--days',nargs='+',type=int,default=[1,2,3,4,5,6,7])
    p.add_argument('--outdir',default='results/dwt_level_analysis')
    main(p.parse_args())
