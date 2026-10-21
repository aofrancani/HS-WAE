"""Post-hoc counterfactual, faithfulness, and robustness analysis for one saved run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from datasets.build_dataloader import build_datamodule
from scripts.run_wtam_definitive import RawTargetDataset, evaluate, make_model


def loader_for(dm, name):
    base = getattr(dm, f"{name}_ds")
    template = dm.test_dataloader()
    return DataLoader(base, batch_size=dm.batch_size, shuffle=False, num_workers=dm.num_workers,
                      pin_memory=dm.pin_memory, collate_fn=template.collate_fn)


def load_run(run_dir, device):
    cfg = __import__('yaml').safe_load((run_dir/'config_resolved.yaml').read_text())
    prov = json.loads((run_dir/'provenance.json').read_text())
    seed_everything(int(prov['seed']), workers=True)
    dm = build_datamodule(cfg)
    if prov['variant'] == 'raw_ae':
        st=dm.standardizer_stats
        for n in ('train_ds','val_ds','test_ds'):
            setattr(dm,n,RawTargetDataset(getattr(dm,n),st['mean'],st['std']))
    model=make_model(cfg,dm,prov['variant'])
    ckpt=torch.load(prov['best_checkpoint'],map_location='cpu')['state_dict']
    model.load_state_dict({k[6:]:v for k,v in ckpt.items() if k.startswith('model.')})
    model.to(device).eval()
    return cfg,prov,dm,model


def compact(m):
    keys=('accuracy','f1','balanced_accuracy','auroc','auprc','nll','brier','recon_mse','recon_sam')
    return {k:m[k] for k in keys}


def true_sample_effects(clean, changed):
    y=clean['y_true']; ix=np.arange(len(y))
    pc=clean['probabilities'][ix,y]; pm=changed['probabilities'][ix,y]
    ce_c=-np.log(np.maximum(pc,1e-12)); ce_m=-np.log(np.maximum(pm,1e-12))
    return ce_m-ce_c, pc-pm


def raw_level_energy(raw,cfg):
    d=cfg['data']['preprocess']['dwt']; rows=[]
    for x in raw:
        c=pywt.wavedec(x,wavelet=d.get('wavelet','db4'),level=d.get('level'),mode=d.get('boundary','periodization'))
        if not d.get('include_approx',True): c=c[1:]
        rows.append([float(np.sum(v*v)) for v in c])
    return np.asarray(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--permutations',type=int,default=30); ap.add_argument('--outdir'); ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args(); run=Path(a.run_dir); out=Path(a.outdir) if a.outdir else run/'analysis'
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True); device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    cfg,prov,dm,model=load_run(run,device); test=dm.test_dataloader(); clean_m,clean=evaluate(model,test,device)
    summary=[{'analysis':'clean','replicate':-1,**compact(clean_m)}]
    if prov['variant']=='wtam_full':
        train_m,train=evaluate(model,loader_for(dm,'train'),device); mean=torch.tensor(train['alpha'].mean(0))
        k=clean['alpha'].shape[1]; uniform=torch.full((k,),1/k)
        for name,alpha in [('uniform',uniform),('train_mean',mean)]:
            m,_=evaluate(model,test,device,alpha_override=alpha); summary.append({'analysis':name,'replicate':-1,**compact(m)})
        rng=np.random.default_rng(20260911)
        nperm=1 if a.smoke else a.permutations
        perm_metrics={'sample_shuffled':[],'level_permuted':[]}
        for i in range(nperm):
            alpha=torch.tensor(clean['alpha'][rng.permutation(len(clean['alpha']))])
            m,_=evaluate(model,test,device,alpha_override=alpha); cm=compact(m); perm_metrics['sample_shuffled'].append(cm); summary.append({'analysis':'sample_shuffled','replicate':i,**cm})
            alpha=torch.tensor(clean['alpha'][:,rng.permutation(k)])
            m,_=evaluate(model,test,device,alpha_override=alpha); cm=compact(m); perm_metrics['level_permuted'].append(cm); summary.append({'analysis':'level_permuted','replicate':i,**cm})
        for name,vals in perm_metrics.items():
            summary.append({'analysis':name+'_mean','replicate':-1,**{k:float(np.mean([v[k] for v in vals])) for k in vals[0]}})
        energy=raw_level_energy(clean['raw_target'],cfg); ce_cols=[]; prob_cols=[]
        levels=range(1 if a.smoke else k)
        for level in levels:
            _,masked=evaluate(model,test,device,corrupt_level=level,corruption='occlusion')
            ce,pr=true_sample_effects(clean,masked); ce_cols.append(ce); prob_cols.append(pr)
        if not a.smoke:
            ce=np.stack(ce_cols,1); pr=np.stack(prob_cols,1); rows=[]
            for i in range(len(clean['y_true'])):
                rows.append({'sample':i,'rho_alpha_ce':spearmanr(clean['alpha'][i],ce[i]).statistic,
                             'rho_alpha_prob':spearmanr(clean['alpha'][i],pr[i]).statistic,
                             'rho_energy_ce':spearmanr(energy[i],ce[i]).statistic,
                             'rho_energy_prob':spearmanr(energy[i],pr[i]).statistic})
            faith=pd.DataFrame(rows); faith.to_csv(out/'faithfulness_sample.csv',index=False)
            (out/'faithfulness_run.json').write_text(json.dumps({k:float(faith[k].mean()) for k in faith.columns if k.startswith('rho_')},indent=2)+'\n')
            np.savez_compressed(out/'faithfulness_arrays.npz',alpha=clean['alpha'],raw_energy=energy,
                                importance_ce=ce,importance_probability=pr,y_true=clean['y_true'])
    if prov['variant']!='raw_ae':
        k=len(dm.dwt_level_lengths); rows=[]; alpha_shift=[]
        levels=range(1 if a.smoke else k)
        conditions=[('level_dropout',0.0),('gaussian',.25)] if a.smoke else [('level_dropout',0.0),('gaussian',.25),('gaussian',.5),('gaussian',1.0),('coefficient_dropout',.25),('coefficient_dropout',.5)]
        for level in levels:
            for kind,sev in conditions:
                torch.manual_seed(20260911+100*level+int(sev*100))
                corr='occlusion' if kind=='level_dropout' else kind
                m,arr=evaluate(model,test,device,corrupt_level=level,corruption=corr,severity=sev)
                row={'level':level,'level_name':prov['dwt_level_names'][level],'corruption':kind,'severity':sev,**compact(m)}
                if prov['variant']=='wtam_full':
                    clean_ent=-np.sum(clean['alpha']*np.log(np.maximum(clean['alpha'],1e-12)),axis=1).mean()
                    corr_ent=-np.sum(arr['alpha']*np.log(np.maximum(arr['alpha'],1e-12)),axis=1).mean()
                    row['alpha_entropy_change']=float(corr_ent-clean_ent)
                row['f1_degradation']=clean_m['f1']-m['f1']; rows.append(row)
                if prov['variant']=='wtam_full': alpha_shift.append({'level':level,'corruption':kind,'severity':sev,'shift':(arr['alpha']-clean['alpha']).mean(0)})
        pd.DataFrame(rows).to_csv(out/'robustness.csv',index=False)
        rob={'mean_f1_degradation':float(np.mean([x['f1_degradation'] for x in rows])),'conditions':len(rows)}
        (out/'robustness_summary.json').write_text(json.dumps(rob,indent=2)+'\n')
        if alpha_shift:
            np.savez_compressed(out/'alpha_shift.npz',level=np.array([x['level'] for x in alpha_shift]),
                corruption=np.array([x['corruption'] for x in alpha_shift]),severity=np.array([x['severity'] for x in alpha_shift]),
                mean_alpha_shift=np.stack([x['shift'] for x in alpha_shift]))
    pd.DataFrame(summary).to_csv(out/'counterfactual_metrics.csv',index=False)
    (out/'analysis_provenance.json').write_text(json.dumps({'source_run':str(run.resolve()),'permutations':1 if a.smoke else a.permutations,'smoke':a.smoke},indent=2)+'\n')
    print(out)
if __name__=='__main__': main()
