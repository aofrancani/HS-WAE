"""Run real module commands on local data, with one-epoch test-only configs.

Run from repository root: python -m tests.run_smoke --outdir /path/to/empty/results
This checks software execution, not numerical reproduction of paper scores.
"""
import argparse, importlib.metadata, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tests.command_runner import run_check, save_report

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--bash')
    ap.add_argument('--hyperblood-raw',type=Path)
    ap.add_argument('--section',choices=['all','imports','shell','training','controlled','baselines','preparation'],default='all')
    a=ap.parse_args()
    bash=a.bash or shutil.which('bash')
    if a.section=='shell' and not bash:
        ap.error('Shell checks require Bash; provide --bash /path/to/bash.')
    if a.section=='preparation' and not a.hyperblood_raw:
        ap.error('Preparation checks require --hyperblood-raw /path/to/HyperBlood.')
    if a.hyperblood_raw and not a.hyperblood_raw.is_dir():
        ap.error('--hyperblood-raw must point to an existing dataset directory.')
    out=a.outdir.resolve(); out.mkdir(parents=True,exist_ok=True)
    report_path=out/('report_'+a.section+'.json')
    if report_path.exists(): ap.error('Use a fresh output directory or a different section.')
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHON=sys.executable.replace('\\','/'))
    report={'python':sys.version,'executable':sys.executable,'versions':{},'checks':[], 'skipped':[], 'completed':False}
    for pkg in ['torch','lightning','numpy','pandas','scipy','scikit-learn','PyWavelets','torchmetrics','xgboost','spectral']:
        try: report['versions'][pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: report['versions'][pkg]='missing'
    def save(): save_report(report_path, report)
    save()
    def run(name,cwd,args,timeout=600):
        return run_check(name,cwd,args,env=env,outdir=out,report_path=report_path,report=report,timeout=timeout)
    def module(name,cwd,mod,*args): return run(name,cwd,[sys.executable,'-m',mod,*map(str,args)])
    def config(project,relative,tag):
        cwd=ROOT/project
        c=yaml.safe_load((cwd/relative).read_text(encoding='utf-8'))
        c['data']['csv_path']=str((cwd/c['data']['csv_path']).resolve())
        c['data']['num_workers']=0
        c['trainer'].update(accelerator='cpu',devices='auto',max_epochs=1,log_every_n_steps=1)
        c['logging']['save_dir']=str(out/'logs'/tag)
        target=out/'configs'/tag/Path(relative).name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
        return target
    contexts=[('soybean','HS-AE','configs/dwt_attn.yaml','main','main_stats'),('soybean_raw','HS-AE/v0','configs/no-dwt_reduced.yaml','main','main_stats'),('blood','HSI_Blood_mc','configs/dwt_attn.yaml','main','main_stats'),('blood_raw','HSI_Blood_mc','v0/configs/no-dwt.yaml','v0.main','v0.main_stats')]
    if a.section in ['all','imports']:
        module('integrity',ROOT,'unittest','discover','-s','tests','-v')
        for context,project,relative,main_mod,stats_mod in contexts:
            run('runtime_'+context,ROOT/project,[sys.executable,str(ROOT/'tests/runtime_probe.py'),'--context',context,'--project-root','.', '--output',str(out/(context+'.json'))])
            for mod in [main_mod,stats_mod]: module('help_'+context+'_'+mod.replace('.','_'),ROOT/project,mod,'--help')
        scripts=[p.stem for p in (ROOT/'HS-AE/scripts').glob('*.py') if p.stem not in ['compute_metrics_stats','finalize_wtam_definitive']]
        for script in scripts: module('help_'+script,ROOT/'HS-AE','scripts.'+script,'--help')
        module('help_main_stats_latent',ROOT/'HS-AE','main_stats_latent','--help')
        for mod in ['preprocessing.hyperblood.data_preparation','preprocessing.hyperblood.generate_fig_paper','tools.configure','tools.prepare_hyperblood']:
            module('help_'+mod.replace('.','_'),ROOT,mod,'--help')
    if a.section in ['all','shell']:
        if bash:
            for script in (ROOT/'scripts').glob('*.sh'): run('bash_syntax_'+script.stem,ROOT,[bash,'-n',script.as_posix()])
            for script,variant in [('run_soybean','hswae'),('run_soybean','raw'),('run_soybean','dwt'),('run_hyperblood','hswae'),('run_hyperblood','raw'),('run_controlled','soybean'),('run_controlled','hyperblood'),('analyze_controlled','soybean')]:
                run('bash_help_'+script+'_'+variant,ROOT,[bash,(ROOT/'scripts'/f'{script}.sh').as_posix(),variant,'--help'])
            previous_dry_run=env.get('DRY_RUN')
            env['DRY_RUN']='1'
            for campaign in ['soybean','hyperblood','controlled-soybean','controlled-hyperblood','baselines','reductions']:
                run('bash_matrix_'+campaign,ROOT,[bash,(ROOT/'scripts/paper_sweep.sh').as_posix(),campaign])
            if previous_dry_run is None: env.pop('DRY_RUN')
            else: env['DRY_RUN']=previous_dry_run
        else: report['skipped'].append('Bash unavailable; shell checks not exercised.'); save()
    if a.section in ['all','training']:
        for context,project,relative,main_mod,stats_mod in contexts:
            cfg=config(project,relative,'single_'+context)
            band='vis' if context.startswith('blood') else 'nir'
            options=['--config',cfg,'--band_region',band]
            if not context.startswith('blood'): options+=['--days','2']
            ok=module('train_'+context,ROOT/project,main_mod,'--mode','train','--max_epochs','1',*options)
            if ok:
                checkpoints=sorted((out/'logs'/('single_'+context)).rglob('*.ckpt'))
                if not checkpoints: raise AssertionError('Training did not save checkpoint: '+context)
                module('reload_'+context,ROOT/project,main_mod,'--mode','test','--ckpt',checkpoints[0],*options)
            cfg=config(project,relative,'repeated_'+context)
            options[1]=cfg
            module('repeated_'+context,ROOT/project,stats_mod,'--mode','train','--max_epochs','1',*options)
        cfg=config('HS-AE','configs/dwt_attn_norecon.yaml','norecon')
        module('train_norecon',ROOT/'HS-AE','main','--mode','train','--config',cfg,'--band_region','nir','--days','2','--max_epochs','1')
        for name in ['dwt_attn','dwt_controlled']:
            cfg=config('HS-AE','configs/'+name+'.yaml','latent_'+name)
            module('latent_'+name,ROOT/'HS-AE','main_stats_latent','--mode','train','--config',cfg,'--band_region','nir','--days','2','--max_epochs','1','--n_exp','2','--latent_out_root',out/'latent')
        module('latent_summary',ROOT/'HS-AE','scripts.analyze_latent_25','--root',out/'latent','--days','2','--dwt-method','dwt_controlled','--expected-runs','2','--bootstrap','200','--outdir',out/'latent_summary')
    if a.section in ['all','controlled']:
        for project,context,band in [('HS-AE','soybean','nir'),('HSI_Blood_mc','blood','vis')]:
            cfg=config(project,'configs/wtam_definitive.yaml','controlled_'+context)
            root=out/'controlled'/context
            for variant in ['raw_ae','dwt_concat','dwt_static_gate','dwt_dynamic_gate','wtam_full']:
                options=['--config',cfg,'--variant',variant,'--band-region',band,'--device','cpu','--seed','2','--exp-idx','0','--outdir',root,'--max-epochs','1']
                if context=='soybean': options+=['--days','2']
                if variant=='wtam_full': options+=['--smoke-interventions']
                ok=module('controlled_'+context+'_'+variant,ROOT/project,'scripts.run_wtam_definitive',*options)
                condition=band+('_days_2' if context=='soybean' else '')
                r=root/variant/condition/'run_00_seed_2'
                if ok:
                    arrays=np.load(r/'test_outputs.npz')
                    for key in ['latent','raw_target','reconstruction']:
                        if not np.isfinite(arrays[key]).all(): raise AssertionError(f'Nonfinite {r}/{key}')
                    if variant in ['dwt_concat','wtam_full']:
                        module('interventions_'+context+'_'+variant,ROOT/project,'scripts.analyze_wtam_definitive','--run-dir',r,'--device','cpu','--permutations','2')
    if a.section in ['all','baselines']:
        cfg=config('HS-AE','configs/dwt_attn.yaml','baselines')
        for feature,models in [('raw',['rf','svm','xgb','mlp','cnn1d']),('pca95',['svm']),('pca64',['svm']),('selectk',['svm'])]:
            module('baseline_'+feature,ROOT/'HS-AE','scripts.run_baselines','--config',cfg,'--band_region','nir','--days','2','--feature_mode',feature,'--models',*models,'--n_exp','1','--seed','2','--device','cpu','--max_epochs','1','--outdir',out/'baselines')
        module('baseline_summary',ROOT/'HS-AE','scripts.analyze_baselines','--root',out/'baselines','--expected-runs','1','--bootstrap','200')
        module('dwt_energy',ROOT/'HS-AE','scripts.analyze_dwt_level_discriminability','--config',cfg,'--band_region','nir','--days','2','--outdir',out/'dwt_energy')
    if a.section in ['all','preparation']:
        if a.hyperblood_raw:
            target=out/'regenerated_B1.csv'
            if module('blood_extract',ROOT,'tools.prepare_hyperblood','--dataset-root',a.hyperblood_raw,'--out',target,'--scenes','B(1)','--min-pixels','60','--connectivity','2','--debug-dir',''):
                old=pd.read_csv(ROOT/'data/HSI_Blood/HSI_Blood.csv'); old=old[old.scene=='B(1)'].reset_index(drop=True)
                new=pd.read_csv(target)
                pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-6,atol=1e-7)
                report['preparation_matches_original']=True
        else: report['skipped'].append('Raw cubes not provided; extraction not exercised.')
    report['passed']=sum(c['exit_code']==0 for c in report['checks'])
    report['failed']=sum(c['exit_code']!=0 for c in report['checks'])
    report['completed']=True
    save(); print(json.dumps({'report':str(report_path),'passed':report['passed'],'failed':report['failed'],'skipped':report['skipped']}),flush=True)
    raise SystemExit(bool(report['failed']))

if __name__=='__main__': main()
