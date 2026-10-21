"""Finalize the pre-specified WTAM study after every required run has completed."""
from __future__ import annotations
import json, shutil, zlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
import matplotlib.pyplot as plt

SOY=Path('/home/users/u9017471/workspace/HS-AE/results/wtam_definitive/soybean')
MC=Path('/home/users/u9017471/workspace/HSI_Blood_mc/results/wtam_definitive/multiclass')
SENS=Path('/home/users/u9017471/workspace/HSI_Blood_mc/results/wtam_definitive/stratified_sensitivity')
BIN=Path('/home/users/u9017471/workspace/HSI_Blood/results/wtam_definitive/binary')
OUT=Path('/home/users/u9017471/workspace/HS-AE/results/wtam_definitive/final_analysis')
CROSS=Path('/home/users/u9017471/workspace/HS-AE/results/wtam_definitive/cross_dataset')
SCENE=Path('/home/users/u9017471/workspace/HSI_Blood_mc/SCENE_SPLIT_FEASIBILITY.md')
EXPECTED={'soybean':875,'multiclass_historical':700,'multiclass_stratified':50,'binary':50}


def run_dirs(root): return sorted(p.parent for p in root.glob('*/*/run_*/metrics.json'))

def meta(run):
    p=json.loads((run/'provenance.json').read_text())
    return {'variant':p['variant'],'seed':int(p['seed']),'exp_idx':int(p['exp_idx']),
            'day':(p.get('days') or [None])[0],'region':p['band_region'],'run_dir':str(run),
            'best_checkpoint':p['best_checkpoint'],'best_epoch':json.loads((run/'metrics.json').read_text())['best_epoch']}

def architecture(root,dataset):
    rows=[]
    for run in run_dirs(root): rows.append({'dataset':dataset,**meta(run),**json.loads((run/'metrics.json').read_text())})
    return pd.DataFrame(rows)

def analysis_table(root,filename):
    rows=[]
    for run in run_dirs(root):
        p=run/'analysis'/filename
        if not p.exists():
            # RAW_AE has no DWT levels, so level analysis is intentionally absent.
            if meta(run)['variant'] == 'raw_ae': continue
            if filename == 'faithfulness_run.json': continue
            raise RuntimeError(f'Missing {p}')
        m=meta(run)
        if filename.endswith('.csv'):
            for row in pd.read_csv(p).to_dict('records'): rows.append({**m,**row})
        else: rows.append({**m,**json.loads(p.read_text())})
    return pd.DataFrame(rows)

def alpha_summaries(root,dataset):
    """Compact run/class alpha summaries from retained test arrays."""
    rows=[]
    for run in run_dirs(root):
        m=meta(run); z=np.load(run/'test_outputs.npz')
        if 'alpha' not in z.files: continue
        alpha=np.asarray(z['alpha'],float); y=np.asarray(z['y_true'])
        prov=json.loads((run/'provenance.json').read_text())
        names=prov.get('dwt_level_names') or [f'level_{i}' for i in range(alpha.shape[1])]
        for cls in ['all',*map(int,np.unique(y))]:
            a=alpha if cls=='all' else alpha[y==cls]
            entropy=-np.sum(a*np.log(np.maximum(a,1e-12)),axis=1)
            for j,name in enumerate(names):
                rng=np.random.default_rng(20260911+zlib.crc32(f'{run}:{cls}:{j}'.encode()))
                boot=a[rng.integers(0,len(a),size=(2000,len(a))),j].mean(1)
                rows.append({'dataset':dataset,**m,'class':cls,'level':j,'level_name':name,
                    'n_samples':len(a),'alpha_mean':float(a[:,j].mean()),
                    'alpha_std':float(a[:,j].std(ddof=1)) if len(a)>1 else float('nan'),
                    'alpha_median':float(np.median(a[:,j])),
                    'alpha_ci_low':float(np.quantile(boot,.025)),
                    'alpha_ci_high':float(np.quantile(boot,.975)),
                    'attention_entropy_mean':float(entropy.mean())})
    return pd.DataFrame(rows)

def reconstruction_by_class(root,dataset):
    rows=[]
    for run in run_dirs(root):
        m=meta(run); z=np.load(run/'test_outputs.npz')
        y=np.asarray(z['y_true']); truth=np.asarray(z['raw_target'],float); pred=np.asarray(z['reconstruction'],float)
        for cls in ['all',*map(int,np.unique(y))]:
            take=np.ones(len(y),bool) if cls=='all' else y==cls
            a,b=pred[take],truth[take]; diff=a-b
            dot=np.sum(a*b,axis=1); den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
            rows.append({'dataset':dataset,**m,'class':cls,'n_samples':int(take.sum()),
                'recon_mse':float(np.mean(diff**2)),'recon_mae':float(np.mean(abs(diff))),
                'recon_sam':float(np.mean(np.arccos(np.clip(dot/np.maximum(den,1e-12),-1,1)))),
                'recon_derivative_rmse':float(np.sqrt(np.mean(np.diff(diff,axis=1)**2)))})
    return pd.DataFrame(rows)

def alpha_shift_table(root,dataset):
    rows=[]
    for run in run_dirs(root):
        m=meta(run); p=run/'analysis'/'alpha_shift.npz'
        if not p.exists(): continue
        z=np.load(p); prov=json.loads((run/'provenance.json').read_text())
        names=prov.get('dwt_level_names') or [f'level_{i}' for i in range(z['mean_alpha_shift'].shape[1])]
        for i in range(len(z['level'])):
            corrupted=int(z['level'][i])
            for output,shift in enumerate(z['mean_alpha_shift'][i]):
                rows.append({'dataset':dataset,**m,'corrupted_level':corrupted,
                    'corrupted_level_name':names[corrupted],'output_level':output,
                    'output_level_name':names[output],'corruption':str(z['corruption'][i]),
                    'severity':float(z['severity'][i]),'mean_alpha_shift':float(shift)})
    return pd.DataFrame(rows)

def faithfulness_level_table(root,dataset):
    rows=[]
    for run in run_dirs(root):
        m=meta(run); p=run/'analysis'/'faithfulness_arrays.npz'
        if not p.exists(): continue
        z=np.load(p); prov=json.loads((run/'provenance.json').read_text())
        names=prov.get('dwt_level_names') or [f'level_{i}' for i in range(z['alpha'].shape[1])]
        for j,name in enumerate(names):
            rows.append({'dataset':dataset,**m,'level':j,'level_name':name,
                'alpha_mean':float(np.mean(z['alpha'][:,j])),
                'causal_importance_ce_mean':float(np.mean(z['importance_ce'][:,j])),
                'causal_importance_probability_mean':float(np.mean(z['importance_probability'][:,j])),
                'raw_energy_mean':float(np.mean(z['raw_energy'][:,j]))})
    return pd.DataFrame(rows)

def paired_stats(values,label):
    x=np.asarray(values,float); x=x[np.isfinite(x)]; n=len(x)
    if n<2: raise RuntimeError(f'{label}: only {n} finite pairs')
    rng=np.random.default_rng(20260911+zlib.crc32(label.encode()))
    means=x[rng.integers(0,n,size=(20000,n))].mean(1)
    if np.all(x==0): w,p=0.0,1.0
    else:
        q=wilcoxon(x,zero_method='wilcox',alternative='two-sided'); w,p=float(q.statistic),float(q.pvalue)
    nz=x[x!=0]; ranks=rankdata(abs(nz)); total=ranks.sum(); rb=float((ranks[nz>0].sum()-ranks[nz<0].sum())/total) if total else 0.0
    return {'comparison':label,'n_pairs':n,'mean_benefit':float(x.mean()),'sd_benefit':float(x.std(ddof=1)),
            'ci_low':float(np.quantile(means,.025)),'ci_high':float(np.quantile(means,.975)),
            'wilcoxon_stat':w,'p_raw':p,'rank_biserial':rb}

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_raw']); m=len(rows); running=0
    for rank,i in enumerate(order):
        adj=min(1.0,(m-rank)*rows[i]['p_raw']); running=max(running,adj); rows[i]['p_holm']=running

def benefit_arch(df,a,b,metric,aggregate_day=False):
    keys=['seed']+(['day'] if aggregate_day else [])
    q=df[df.variant.isin([a,b])].pivot(index=keys,columns='variant',values=metric).dropna()
    d=q[b]-q[a]
    return d.groupby(level='seed').mean().to_numpy() if aggregate_day else d.to_numpy()

def architecture_benefit(df,a,b,metric,units=(),lower_is_better=False):
    keys=['seed',*units]
    q=df[df.variant.isin([a,b])].pivot(index=keys,columns='variant',values=metric).dropna()
    d=(q[a]-q[b]) if lower_is_better else (q[b]-q[a])
    return d.groupby(level='seed').mean().to_numpy() if units else d.to_numpy()

def benefit_rob(df,aggregate_day=False):
    keys=['seed']+(['day'] if aggregate_day else [])
    q=df[df.variant.isin(['dwt_concat','wtam_full'])].pivot(index=keys,columns='variant',values='mean_f1_degradation').dropna()
    d=q.dwt_concat-q.wtam_full
    return d.groupby(level='seed').mean().to_numpy() if aggregate_day else d.to_numpy()

def counterfactual_benefit(df,condition,metric,units=()):
    q=df[df.variant=='wtam_full']; keys=['seed',*units]
    p=q[q.analysis.isin(['clean',condition])].pivot(index=keys,columns='analysis',values=metric).dropna()
    lower=metric in ('nll','brier','recon_mse','recon_sam')
    d=(p[condition]-p.clean) if lower else (p.clean-p[condition])
    return d.groupby(level='seed').mean().to_numpy() if units else d.to_numpy()

def robustness_benefit(df,metric,units=()):
    keys=['seed',*units,'variant']
    q=df[df.variant.isin(['dwt_concat','wtam_full'])].groupby(keys)['degradation_'+metric].mean().unstack('variant').dropna()
    d=q['dwt_concat']-q['wtam_full']
    return d.groupby(level='seed').mean().to_numpy() if units else d.to_numpy()

def benefit_cf(df,condition,metric,aggregate_day=False):
    return counterfactual_benefit(df,condition,metric,('day',) if aggregate_day else ())

def faithfulness_benefit(df,importance='ce',versus_energy=False,units=()):
    keys=['seed',*units]; q=df.set_index(keys)
    alpha=q['rho_alpha_'+importance]
    d=alpha-q['rho_energy_'+importance] if versus_energy else alpha
    return d.groupby(level='seed').mean().to_numpy() if units else d.to_numpy()

def benefit_faith(df,kind,aggregate_day=False):
    return faithfulness_benefit(df,'ce',kind!='positive',('day',) if aggregate_day else ())

def positive(r): return r['mean_benefit']>0 and r['ci_low']>0 and r['p_holm']<.05

def adverse(r): return r['mean_benefit']<0 and r['ci_high']<0 and r['p_holm']<.05


def main():
    roots={'soybean':SOY,'multiclass_historical':MC,'multiclass_stratified':SENS,'binary':BIN}
    counts={k:len(run_dirs(v)) for k,v in roots.items()}
    if counts!=EXPECTED: raise SystemExit(f'Campaign incomplete: {counts}; expected {EXPECTED}')
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError(f'Refusing to overwrite {OUT}')
    if CROSS.exists() and any(CROSS.iterdir()): raise FileExistsError(f'Refusing to overwrite {CROSS}')
    OUT.mkdir(parents=True)
    arch=pd.concat([architecture(v,k) for k,v in roots.items()],ignore_index=True)
    arch.to_csv(OUT/'architecture_results.csv',index=False)
    arch[['dataset','variant','region','day','seed','recon_mse','recon_mae','recon_sam','recon_derivative_rmse']].to_csv(OUT/'reconstruction_results.csv',index=False)
    arch[arch.dataset=='multiclass_stratified'].to_csv(OUT/'multiclass_stratified_sensitivity_results.csv',index=False)
    arch[arch.dataset=='binary'].to_csv(OUT/'binary_b1_b4_results.csv',index=False)

    alpha=pd.concat([alpha_summaries(v,k) for k,v in roots.items()],ignore_index=True)
    alpha.to_csv(OUT/'alpha_run_class_level_summary.csv',index=False)
    recon_class=pd.concat([reconstruction_by_class(v,k) for k,v in roots.items()],ignore_index=True)
    recon_class.to_csv(OUT/'reconstruction_by_class_results.csv',index=False)
    shifts=pd.concat([alpha_shift_table(v,k) for k,v in roots.items()],ignore_index=True)
    shifts.to_csv(OUT/'adaptive_alpha_shift_results.csv',index=False)
    faith_level=pd.concat([faithfulness_level_table(v,k) for k,v in roots.items()],ignore_index=True)
    faith_level.to_csv(OUT/'attention_faithfulness_by_level.csv',index=False)

    soy_rob=analysis_table(SOY,'robustness_summary.json'); mc_rob=analysis_table(MC,'robustness_summary.json')
    sens_rob=analysis_table(SENS,'robustness_summary.json'); bin_rob=analysis_table(BIN,'robustness_summary.json')
    pd.concat([soy_rob.assign(dataset='soybean'),mc_rob.assign(dataset='multiclass_historical'),sens_rob.assign(dataset='multiclass_stratified'),bin_rob.assign(dataset='binary')]).to_csv(OUT/'robustness_run_summary.csv',index=False)
    robustness=[]
    for root,dataset in [(SOY,'soybean'),(MC,'multiclass_historical'),(SENS,'multiclass_stratified'),(BIN,'binary')]:
        z=analysis_table(root,'robustness.csv'); z['dataset']=dataset; robustness.append(z)
    robustness=pd.concat(robustness,ignore_index=True)
    def clean_lookup(row,metric):
        day_match=arch.day.isna() if pd.isna(row.day) else arch.day.eq(row.day)
        q=arch[(arch.dataset==row.dataset)&(arch.variant==row.variant)&(arch.region==row.region)&
               (arch.seed==row.seed)&day_match]
        if len(q)!=1: raise RuntimeError(f'Clean-result join is not one-to-one for {row.dataset}/{row.variant}/{row.region}/{row.seed}/{row.day}')
        return float(q.iloc[0][metric])
    for metric in ('accuracy','f1','auroc','nll','brier'):
        robustness['clean_'+metric]=robustness.apply(lambda r:clean_lookup(r,metric),axis=1)
        robustness['degradation_'+metric]=(robustness['clean_'+metric]-robustness[metric]) if metric!='nll' else (robustness[metric]-robustness['clean_'+metric])
    robustness.to_csv(OUT/'robustness_results.csv',index=False)

    soy_cf=analysis_table(SOY,'counterfactual_metrics.csv'); mc_cf=analysis_table(MC,'counterfactual_metrics.csv')
    sens_cf=analysis_table(SENS,'counterfactual_metrics.csv'); bin_cf=analysis_table(BIN,'counterfactual_metrics.csv')
    pd.concat([soy_cf.assign(dataset='soybean'),mc_cf.assign(dataset='multiclass_historical'),sens_cf.assign(dataset='multiclass_stratified'),bin_cf.assign(dataset='binary')]).to_csv(OUT/'counterfactual_results.csv',index=False)
    soy_f=analysis_table(SOY,'faithfulness_run.json'); mc_f=analysis_table(MC,'faithfulness_run.json')
    sens_f=analysis_table(SENS,'faithfulness_run.json'); bin_f=analysis_table(BIN,'faithfulness_run.json')
    pd.concat([soy_f.assign(dataset='soybean'),mc_f.assign(dataset='multiclass_historical'),sens_f.assign(dataset='multiclass_stratified'),bin_f.assign(dataset='binary')]).to_csv(OUT/'attention_faithfulness_results.csv',index=False)

    primary=[]
    h1=[paired_stats(benefit_rob(soy_rob,True),'H1 soybean: concat minus WTAM F1 degradation'),
        paired_stats(benefit_rob(mc_rob[mc_rob.region=='all']),'H1 multiclass all: concat minus WTAM F1 degradation')]
    for r in h1:r['hypothesis']='H1'
    holm(h1); primary+=h1
    h2=[]
    for name,df,agg in [('soybean',soy_cf,True),('multiclass all',mc_cf[mc_cf.region=='all'],False)]:
        for cond in ('uniform','sample_shuffled_mean'):
            for metric in ('nll','auroc'):
                h2.append({**paired_stats(benefit_cf(df,cond,metric,agg),f'H2 {name}: learned vs {cond} {metric}'),'hypothesis':'H2'})
    holm(h2);primary+=h2
    h3=[]
    for name,df,agg in [('soybean',soy_f,True),('multiclass all',mc_f[mc_f.region=='all'],False)]:
        h3.append({**paired_stats(benefit_faith(df,'positive',agg),f'H3 {name}: alpha-importance rho > 0'),'hypothesis':'H3'})
        h3.append({**paired_stats(benefit_faith(df,'vs_energy',agg),f'H3 {name}: alpha rho minus energy rho'),'hypothesis':'H3'})
    holm(h3);primary+=h3
    mc_all=arch[(arch.dataset=='multiclass_historical')&(arch.region=='all')]
    h4=[{**paired_stats(benefit_arch(mc_all,'dwt_concat','wtam_full','f1'), 'H4 multiclass all: WTAM minus concat macro F1'),'hypothesis':'H4'},
        {**paired_stats(benefit_arch(mc_all,'dwt_concat','wtam_full','latent_normalized_separation'),'H4 multiclass all: WTAM minus concat normalized separation'),'hypothesis':'H4'}]
    holm(h4);primary+=h4
    groups={'H1':h1,'H2':h2,'H3':h3,'H4':h4}; status={}
    status['H1']='SUPPORTED' if all(map(positive,h1)) else ('PARTIALLY SUPPORTED' if any(map(positive,h1)) else 'NOT SUPPORTED')
    status['H2']='SUPPORTED' if all(map(positive,h2)) else ('PARTIALLY SUPPORTED' if any(map(positive,h2)) else 'NOT SUPPORTED')
    status['H3']='SUPPORTED' if all(map(positive,h3)) else ('PARTIALLY SUPPORTED' if any(map(positive,h3)) else 'NOT SUPPORTED')
    status['H4']='SUPPORTED' if any(map(positive,h4)) and not any(map(adverse,h4)) else ('PARTIALLY SUPPORTED' if any(r['mean_benefit']>0 for r in h4) else 'NOT SUPPORTED')
    for r in primary:r['status']=status[r['hypothesis']]
    pd.DataFrame(primary).to_csv(OUT/'final_wtam_primary_hypotheses.csv',index=False)

    robustness_tests=[]
    for dataset,df,units in [
        ('soybean',robustness[robustness.dataset=='soybean'],('day',)),
        ('multiclass all',robustness[(robustness.dataset=='multiclass_historical')&(robustness.region=='all')],()),
        ('multiclass all-regions aggregate',robustness[robustness.dataset=='multiclass_historical'],('region',)),
        ('multiclass stratified',robustness[robustness.dataset=='multiclass_stratified'],()),
        ('binary',robustness[robustness.dataset=='binary'],())]:
        fam=[]
        for metric in ('accuracy','f1','auroc','nll','brier'):
            fam.append({**paired_stats(robustness_benefit(df,metric,units),f'Robustness {dataset}: concat minus WTAM {metric} degradation'),
                        'dataset':dataset,'metric':metric})
        holm(fam); robustness_tests += fam
    pd.DataFrame(robustness_tests).to_csv(OUT/'robustness_paired_tests.csv',index=False)

    robustness_condition=[]
    for dataset,base,units in [
        ('soybean',robustness[robustness.dataset=='soybean'],('day',)),
        ('multiclass all',robustness[(robustness.dataset=='multiclass_historical')&(robustness.region=='all')],())]:
        for metric in ('accuracy','f1','auroc','nll','brier'):
            fam=[]
            for (corr,sev),df in base.groupby(['corruption','severity']):
                fam.append({**paired_stats(robustness_benefit(df,metric,units),f'Robustness {dataset} {corr} {sev}: concat minus WTAM {metric} degradation'),
                            'dataset':dataset,'corruption':corr,'severity':sev,'metric':metric})
            holm(fam); robustness_condition += fam
    pd.DataFrame(robustness_condition).to_csv(OUT/'robustness_condition_paired_tests.csv',index=False)

    faith_tests=[]
    for dataset,df,units in [
        ('soybean',soy_f,('day',)),('multiclass all',mc_f[mc_f.region=='all'],()),
        ('multiclass all-regions aggregate',mc_f,('region',)),
        ('multiclass stratified',sens_f,()),('binary',bin_f,())]:
        fam=[]
        for importance in ('ce','prob'):
            for versus in (False,True):
                endpoint=('alpha minus energy rho' if versus else 'alpha rho')+' '+importance
                fam.append({**paired_stats(faithfulness_benefit(df,importance,versus,units),f'Faithfulness {dataset}: {endpoint}'),
                            'dataset':dataset,'importance':importance,'versus_energy':versus})
        holm(fam); faith_tests += fam
    pd.DataFrame(faith_tests).to_csv(OUT/'attention_faithfulness_paired_tests.csv',index=False)

    # Secondary architectural questions are fixed comparisons. They are kept
    # separate from H1--H4 and corrected within dataset-level families.
    higher=['accuracy','f1','auroc','latent_silhouette','latent_fisher','latent_normalized_separation']
    lower=['nll','latent_davies_bouldin','recon_mse','recon_mae','recon_sam','recon_derivative_rmse']
    secondary=[]
    families=[]
    soy_arch=arch[arch.dataset=='soybean']
    fam=[]
    for a,b,name in [('raw_ae','dwt_concat','M0 to M1'),('dwt_concat','dwt_static_gate','M1 to M2'),
                     ('dwt_static_gate','dwt_dynamic_gate','M2 to M3'),('dwt_dynamic_gate','wtam_full','M3 to M4'),
                     ('dwt_concat','wtam_full','M1 to M4')]:
        for metric in higher+lower:
            fam.append({**paired_stats(architecture_benefit(soy_arch,a,b,metric,('day',),metric in lower),
                        f'Secondary soybean {name}: {metric}'),'family':'soybean architecture','metric':metric,'from_variant':a,'to_variant':b})
    families.append(fam)
    mc_arch=arch[arch.dataset=='multiclass_historical']; fam=[]
    for a,b,name in [('dwt_concat','dwt_static_gate','BM1 to BM2'),('dwt_static_gate','dwt_dynamic_gate','BM2 to BM3'),
                     ('dwt_dynamic_gate','wtam_full','BM3 to BM4'),('dwt_concat','wtam_full','BM1 to BM4')]:
        for metric in higher+lower:
            fam.append({**paired_stats(architecture_benefit(mc_arch[mc_arch.region=='all'],a,b,metric,(),metric in lower),
                        f'Secondary multiclass all {name}: {metric}'),'family':'multiclass all architecture','metric':metric,'from_variant':a,'to_variant':b})
    families.append(fam)
    fam=[]
    for dataset,df in [('multiclass stratified',arch[arch.dataset=='multiclass_stratified']),('binary',arch[arch.dataset=='binary'])]:
        for metric in higher+lower:
            fam.append({**paired_stats(architecture_benefit(df,'dwt_concat','wtam_full',metric,(),metric in lower),
                        f'Secondary {dataset} M1 to M4: {metric}'),'family':dataset+' architecture','metric':metric,
                        'from_variant':'dwt_concat','to_variant':'wtam_full'})
        holm(fam); secondary += fam; fam=[]
    for fam in families:
        holm(fam); secondary += fam
    pd.DataFrame(secondary).to_csv(OUT/'secondary_architecture_paired_tests.csv',index=False)

    cf_tests=[]
    for dataset,df,units in [
        ('soybean',soy_cf,('day',)),('multiclass all',mc_cf[mc_cf.region=='all'],()),
        ('multiclass all-regions aggregate',mc_cf,('region',)),
        ('multiclass stratified',sens_cf,()),('binary',bin_cf,())]:
        fam=[]
        for cond in ('uniform','train_mean','sample_shuffled_mean','level_permuted_mean'):
            for metric in ('accuracy','f1','auroc','nll','brier'):
                fam.append({**paired_stats(counterfactual_benefit(df,cond,metric,units),
                            f'Counterfactual {dataset}: learned vs {cond} {metric}'),
                            'dataset':dataset,'condition':cond,'metric':metric})
        holm(fam); cf_tests += fam
    pd.DataFrame(cf_tests).to_csv(OUT/'counterfactual_paired_tests.csv',index=False)

    cf_condition=[]
    for cond in ('uniform','train_mean','sample_shuffled_mean','level_permuted_mean'):
        for metric in ('accuracy','f1','auroc','nll','brier'):
            fam=[]
            for day in range(1,8):
                df=soy_cf[soy_cf.day==day]
                fam.append({**paired_stats(counterfactual_benefit(df,cond,metric),f'Counterfactual soybean day {day}: learned vs {cond} {metric}'),
                            'dataset':'soybean','condition_name':f'day_{day}','intervention':cond,'metric':metric})
            holm(fam); cf_condition += fam
            fam=[]
            for region in ['all','vis','vis-nir','rgb','re','nir','re-nir']:
                df=mc_cf[mc_cf.region==region]
                fam.append({**paired_stats(counterfactual_benefit(df,cond,metric),f'Counterfactual multiclass {region}: learned vs {cond} {metric}'),
                            'dataset':'multiclass_historical','condition_name':region,'intervention':cond,'metric':metric})
            holm(fam); cf_condition += fam
    pd.DataFrame(cf_condition).to_csv(OUT/'counterfactual_condition_paired_tests.csv',index=False)

    recon_tests=[]
    soy_recon=recon_class[recon_class.dataset=='soybean']
    for metric in ('recon_mse','recon_mae','recon_sam','recon_derivative_rmse'):
        fam=[]
        for cls in ('all',0,1):
            df=soy_recon[soy_recon['class'].astype(str)==str(cls)]
            fam.append({**paired_stats(architecture_benefit(df,'dwt_concat','wtam_full',metric,('day',),True),
                        f'Reconstruction soybean class {cls}: WTAM vs concat {metric}'),
                        'dataset':'soybean','class':cls,'metric':metric})
        holm(fam); recon_tests += fam
    pd.DataFrame(recon_tests).to_csv(OUT/'reconstruction_by_class_paired_tests.csv',index=False)

    # Condition-level M1-vs-M4 estimates are descriptive/supplementary; each
    # day or region still contains 25 paired seeds and Holm is per endpoint.
    condition=[]
    for metric in ['accuracy','f1','auroc','nll','latent_normalized_separation']:
        fam=[]
        for day in range(1,8):
            df=soy_arch[soy_arch.day==day]
            fam.append({**paired_stats(architecture_benefit(df,'dwt_concat','wtam_full',metric,(),metric=='nll'),
                        f'Soybean day {day} M1 to M4: {metric}'),'dataset':'soybean','condition':f'day_{day}','metric':metric})
        holm(fam); condition += fam
        fam=[]
        for region in ['all','vis','vis-nir','rgb','re','nir','re-nir']:
            df=mc_arch[mc_arch.region==region]
            fam.append({**paired_stats(architecture_benefit(df,'dwt_concat','wtam_full',metric,(),metric=='nll'),
                        f'Multiclass {region} BM1 to BM4: {metric}'),'dataset':'multiclass_historical','condition':region,'metric':metric})
        holm(fam); condition += fam
    pd.DataFrame(condition).to_csv(OUT/'condition_level_paired_tests.csv',index=False)

    adaptive=[]
    for dataset,source,unit_df,agg_days in [
        ('soybean','soybean',shifts[shifts.dataset=='soybean'],True),
        ('multiclass all','multiclass_historical',shifts[(shifts.dataset=='multiclass_historical')&(shifts.region=='all')],False),
        ('multiclass stratified','multiclass_stratified',shifts[shifts.dataset=='multiclass_stratified'],False),
        ('binary','binary',shifts[shifts.dataset=='binary'],False)]:
        fam=[]; diagonal=unit_df[unit_df.output_level==unit_df.corrupted_level]
        for (corr,sev),g in diagonal.groupby(['corruption','severity']):
            units=['seed','day'] if agg_days else ['seed']
            x=-g.groupby(units).mean(numeric_only=True)['mean_alpha_shift']
            if agg_days: x=x.groupby(level='seed').mean()
            fam.append({**paired_stats(x.to_numpy(),f'Adaptive {dataset}: reduction on corrupted level, {corr} {sev}'),
                        'dataset':dataset,'corruption':corr,'severity':sev,'endpoint':'negative_corrupted_level_alpha_shift'})
        rq=robustness[(robustness.dataset==source)&(robustness.variant=='wtam_full')]
        if source=='multiclass_historical': rq=rq[rq.region=='all']
        for (corr,sev),g in rq.groupby(['corruption','severity']):
            units=['seed','day'] if agg_days else ['seed']
            x=g.groupby(units).mean(numeric_only=True)['alpha_entropy_change']
            if agg_days: x=x.groupby(level='seed').mean()
            fam.append({**paired_stats(x.to_numpy(),f'Adaptive {dataset}: attention entropy change, {corr} {sev}'),
                        'dataset':dataset,'corruption':corr,'severity':sev,'endpoint':'attention_entropy_change'})
        holm(fam); adaptive += fam
    pd.DataFrame(adaptive).to_csv(OUT/'adaptive_alpha_response_stats.csv',index=False)

    # Compact mean/SD summaries retain every condition rather than selecting favorable rows.
    summary=arch.groupby(['dataset','variant','region','day'],dropna=False).agg({m:['mean','std'] for m in ['accuracy','f1','balanced_accuracy','auroc','auprc','nll','brier','latent_silhouette','latent_davies_bouldin','latent_fisher','latent_normalized_separation','recon_mse','recon_mae','recon_sam','recon_derivative_rmse']}).reset_index()
    summary.columns=['_'.join(str(x) for x in c if str(x)) for c in summary.columns]
    summary.to_csv(OUT/'architecture_mean_sd.csv',index=False)
    shutil.copy2(SCENE,OUT/'SCENE_SPLIT_FEASIBILITY.md')

    fig,ax=plt.subplots(figsize=(9,max(5,.38*len(primary))))
    y=np.arange(len(primary)); means=np.array([r['mean_benefit'] for r in primary]); lo=means-np.array([r['ci_low'] for r in primary]); hi=np.array([r['ci_high'] for r in primary])-means
    ax.errorbar(means,y,xerr=[lo,hi],fmt='o',capsize=3);ax.axvline(0,color='black',lw=1);ax.set_yticks(y,[r['comparison'] for r in primary],fontsize=7);ax.invert_yaxis();ax.set_xlabel('Paired benefit (positive favors pre-specified WTAM hypothesis)');fig.tight_layout();fig.savefig(OUT/'primary_hypothesis_effects.png',dpi=300);fig.savefig(OUT/'primary_hypothesis_effects.pdf');plt.close(fig)

    # Publication figures use complete pre-specified conditions and run-level
    # summaries; no favorable day, level, region, or severity is selected.
    q=alpha[(alpha.dataset=='soybean')&(alpha.variant=='wtam_full')&(alpha['class'].astype(str)=='all')]
    g=q.groupby(['day','level_name']).alpha_mean.mean().unstack()
    fig,ax=plt.subplots(figsize=(6.5,4));
    for col in g: ax.plot(g.index,g[col],marker='o',label=col)
    ax.set(xlabel='Acquisition day',ylabel='Mean WTAM alpha');ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(OUT/'wtam_alpha_by_day.pdf');fig.savefig(OUT/'wtam_alpha_by_day.png',dpi=300);plt.close(fig)

    q=alpha[(alpha.dataset=='soybean')&(alpha.variant=='wtam_full')&(alpha['class'].astype(str)!='all')]
    g=q.groupby(['class','level_name']).alpha_mean.mean().unstack()
    fig,ax=plt.subplots(figsize=(6.5,4));x=np.arange(len(g.columns));w=.8/max(len(g),1)
    for i,(cls,row) in enumerate(g.iterrows()): ax.bar(x+(i-(len(g)-1)/2)*w,row.to_numpy(),w,label=f'class {cls}')
    ax.set_xticks(x,g.columns);ax.set(ylabel='Mean WTAM alpha',xlabel='DWT level');ax.legend();fig.tight_layout();fig.savefig(OUT/'wtam_alpha_by_class.pdf');fig.savefig(OUT/'wtam_alpha_by_class.png',dpi=300);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,(dataset,title) in zip(axes,[('soybean','Soybean'),('multiclass_historical','Multiclass blood (all)')]):
        q=faith_level[faith_level.dataset==dataset]
        if dataset=='multiclass_historical': q=q[q.region=='all']
        g=q.groupby('level_name')[['alpha_mean','causal_importance_ce_mean']].mean()
        ax.scatter(g.alpha_mean,g.causal_importance_ce_mean)
        for name,row in g.iterrows(): ax.annotate(name,(row.alpha_mean,row.causal_importance_ce_mean),fontsize=7)
        ax.set(xlabel='Mean alpha',ylabel='Mean CE occlusion importance',title=title)
    fig.tight_layout();fig.savefig(OUT/'wtam_alpha_vs_occlusion_importance.pdf');fig.savefig(OUT/'wtam_alpha_vs_occlusion_importance.png',dpi=300);plt.close(fig)

    for metric,outname in [('f1','wtam_robustness_f1'),('auroc','wtam_robustness_auroc')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,(dataset,title) in zip(axes,[('soybean','Soybean'),('multiclass_historical','Multiclass blood (all)')]):
            q=robustness[(robustness.dataset==dataset)&robustness.variant.isin(['dwt_concat','wtam_full'])]
            if dataset=='multiclass_historical': q=q[q.region=='all']
            g=q.groupby(['variant','corruption','severity'])['degradation_'+metric].mean()
            labels=sorted({(b,c) for _,b,c in g.index});x=np.arange(len(labels))
            for variant,mark in [('dwt_concat','o'),('wtam_full','s')]:
                vals=[g.get((variant,c,v),np.nan) for c,v in labels]
                ax.plot(x,vals,marker=mark,label=variant)
            ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[f'{c}\n{s:g}' for c,s in labels],rotation=45,ha='right',fontsize=7)
            ax.set(title=title,ylabel=f'Clean minus corrupted {metric.upper()}');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(OUT/(outname+'.pdf'));fig.savefig(OUT/(outname+'.png'),dpi=300);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(dataset,title) in zip(axes,[('soybean','Soybean'),('multiclass_historical','Multiclass blood (all)')]):
        q=shifts[(shifts.dataset==dataset)&(shifts.corruption=='level_dropout')]
        if dataset=='multiclass_historical': q=q[q.region=='all']
        g=q.groupby(['corrupted_level_name','output_level_name']).mean_alpha_shift.mean().unstack()
        im=ax.imshow(g.to_numpy(),aspect='auto',cmap='coolwarm',vmin=-np.nanmax(abs(g.to_numpy())),vmax=np.nanmax(abs(g.to_numpy())))
        ax.set_xticks(range(len(g.columns)),g.columns,rotation=45,ha='right');ax.set_yticks(range(len(g.index)),g.index)
        ax.set(xlabel='WTAM output level',ylabel='Dropped input level',title=title);fig.colorbar(im,ax=ax,fraction=.046)
    fig.tight_layout();fig.savefig(OUT/'wtam_corruption_alpha_shift.pdf');fig.savefig(OUT/'wtam_corruption_alpha_shift.png',dpi=300);plt.close(fig)

    q=arch[arch.variant.isin(['dwt_concat','wtam_full'])]
    g=q.groupby(['dataset','variant']).recon_mse.agg(['mean','std']).reset_index()
    datasets=list(g.dataset.unique());fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(datasets));w=.35
    for i,variant in enumerate(['dwt_concat','wtam_full']):
        z=g[g.variant==variant].set_index('dataset').reindex(datasets)
        ax.bar(x+(i-.5)*w,z['mean'],w,yerr=z['std'],capsize=3,label=variant)
    ax.set_xticks(x,datasets,rotation=20,ha='right');ax.set(ylabel='Test reconstruction MSE');ax.legend();fig.tight_layout();fig.savefig(OUT/'wtam_reconstruction_comparison.pdf');fig.savefig(OUT/'wtam_reconstruction_comparison.png',dpi=300);plt.close(fig)

    lines=['# Final definitive WTAM report','',f'Completed run counts: `{counts}`. Smoke results were excluded from all inference.','', '## Primary hypotheses','']
    for h in ('H1','H2','H3','H4'):
        lines += [f'- **{h}: {status[h]}**']
    lines += ['','## Pre-specified paired tests','', '| Hypothesis | Comparison | Mean benefit | 95% CI | Holm p | Rank-biserial |','|---|---|---:|---:|---:|---:|']
    for r in primary: lines.append(f"| {r['hypothesis']} | {r['comparison']} | {r['mean_benefit']:.6g} | [{r['ci_low']:.6g}, {r['ci_high']:.6g}] | {r['p_holm']:.4g} | {r['rank_biserial']:.4f} |")
    lines += ['','Soybean temporal effects were averaged across seven days within each seed before testing. The multiclass `all` region is primary; supplementary-region effects are not substituted for it and regions are not treated as independent. The fixed-stratified sensitivity and binary blood results remain separate from the historical multiclass analysis.','','A directional result is counted as supporting evidence only when the paired bootstrap interval excludes zero in the predicted direction and the within-family Holm-adjusted Wilcoxon p-value is below .05. Smoke and superseded initialization-order runs are excluded.','','The accompanying CSV files contain every clean, latent, reconstruction, counterfactual, corruption, adaptive-alpha, and faithfulness endpoint. Secondary architecture transitions and per-day/per-region estimates are labeled separately. Null, mixed, and adverse results are retained.']
    report=OUT/'FINAL_WTAM_REPORT.md'; report.write_text('\n'.join(lines)+'\n')
    manifest={'counts':counts,'expected':EXPECTED,'hypothesis_status':status,
              'primary_multiclass_region':'all','soybean_inference_unit':'seed after averaging seven days',
              'smoke_included':False,'superseded_runs_included':False}
    (OUT/'completion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    # Preserve the originally requested cross-dataset filenames as compact
    # aliases; checkpoints remain in their source repositories.
    CROSS.mkdir(parents=True)
    shutil.copy2(OUT/'architecture_mean_sd.csv',CROSS/'cross_dataset_wtam_summary.csv')
    shutil.copy2(OUT/'final_wtam_primary_hypotheses.csv',CROSS/'cross_dataset_primary_hypotheses.csv')
    shutil.copy2(report,CROSS/'cross_dataset_wtam_report.md')
    for name in ['primary_hypothesis_effects.pdf','wtam_robustness_f1.pdf','wtam_robustness_auroc.pdf','wtam_reconstruction_comparison.pdf']:
        shutil.copy2(OUT/name,CROSS/name)
    print(json.dumps({'output':str(OUT),'cross_dataset':str(CROSS),'status':status,'counts':counts},indent=2))
if __name__=='__main__':main()
