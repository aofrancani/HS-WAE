"""Subprocess probe; each project gets its own Python import namespace."""
from pathlib import Path
import argparse, copy, importlib, json, sys
import numpy as np
import torch
import yaml

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--context',choices=['soybean','soybean_raw','blood','blood_raw'],required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    root=Path(a.project_root).resolve()
    sys.path.insert(0,str(root))
    torch.set_num_threads(2)
    raw=a.context.endswith('_raw'); blood=a.context.startswith('blood')
    prefix='v0.' if a.context=='blood_raw' else ''
    imported=[]
    for f in sorted(root.rglob('*.py')):
        rel=f.relative_to(root)
        if 'tests' in rel.parts or '__pycache__' in rel.parts: continue
        if not prefix and 'v0' in rel.parts: continue
        if prefix and rel.parts[0]!='v0': continue
        module='.'.join(rel.with_suffix('').parts)
        if module.endswith('.__init__'): module=module[:-9]
        importlib.import_module(module); imported.append(module)
    build_dm=importlib.import_module(prefix+'datasets.build_dataloader').build_datamodule
    build_model=importlib.import_module(prefix+'models.build_model').build_model_and_lit
    utils=importlib.import_module(prefix+'utils')
    configs=sorted((root/('v0/configs' if prefix else 'configs')).glob('*.yaml'))
    checked=[]
    for config in configs:
        cfg=yaml.safe_load(config.read_text(encoding='utf-8'))
        cfg['data']['num_workers']=0
        cfg['data']['persistent_workers']=False
        utils.override_band_region(cfg,'vis' if blood else 'nir')
        if not blood: utils.override_date_filters(cfg,'2')
        dm=build_dm(cfg)
        kwargs={} if raw else {'dm':dm}
        model,lit=build_model(cfg,dm.input_dim,dm.num_classes,**kwargs)
        batch=next(iter(dm.train_dataloader()))
        if len(batch)==3:
            target,x,y=batch
            out=model(x,y,x_recon_target=target)
        else:
            x,y=batch; out=model(x,y)
        assert out['logits'].shape==(len(y),7 if blood else 2)
        assert out['latent'].shape[1]==64
        assert torch.isfinite(out['loss'])
        optim=torch.optim.AdamW(model.parameters(),lr=.001)
        optim.zero_grad(); out['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optim.step()
        if 'level_alpha' in out:
            torch.testing.assert_close(out['level_alpha'].sum(1),torch.ones(len(y)))
        labels=[ds.y.numpy().copy() for ds in [dm.train_ds,dm.val_ds,dm.test_ds]]
        cfg2=copy.deepcopy(cfg); cfg2['seed']=26
        other=build_dm(cfg2)
        for old,ds in zip(labels,[other.train_ds,other.val_ds,other.test_ds]):
            np.testing.assert_array_equal(old,ds.y.numpy())
        checked.append({'config':config.relative_to(root).as_posix(),'classes':dm.num_classes,'bands':len(dm.kept_wavelengths_nm),'input_dim':dm.input_dim,'split_sizes':[len(x) for x in [dm.train_ds,dm.val_ds,dm.test_ds]],'test_labels':np.bincount(dm.test_ds.y.numpy()).tolist(),'loss':float(out['loss'].detach())})
    # Exercise every paper region and day with the historical/preprocessing code.
    cfg=yaml.safe_load(configs[0].read_text(encoding='utf-8')); cfg['data']['num_workers']=0
    regions=['vis','rgb','re','nir'] if blood else ['uv','vis','rgb','re','nir','swir-1','swir-2a','swir-2b','nir-swir-1','re-nir','re-swir-1','vis-nir','vis-re','vis-swir-1']
    region_checks=[]
    for region in regions:
        local=copy.deepcopy(cfg); utils.override_band_region(local,region)
        if not blood: utils.override_date_filters(local,'2')
        dm=build_dm(local)
        assert dm.input_dim>0 and len(dm.kept_wavelengths_nm)>0 and dm.num_classes==(7 if blood else 2)
        region_checks.append([region,len(dm.kept_wavelengths_nm),dm.input_dim])
    days=[]
    if not blood:
        for day in range(1,8):
            local=copy.deepcopy(cfg); utils.override_band_region(local,'nir'); utils.override_date_filters(local,str(day))
            dm=build_dm(local); days.append(sum(len(x) for x in [dm.train_ds,dm.val_ds,dm.test_ds]))
        assert days==[223,175,175,140,216,100,190],days
    if not raw:
        definitive=importlib.import_module('scripts.run_wtam_definitive')
        cfg=yaml.safe_load((root/'configs/wtam_definitive.yaml').read_text()); cfg['data']['num_workers']=0
        utils.override_band_region(cfg,'vis' if blood else 'nir')
        if not blood: utils.override_date_filters(cfg,'2')
        dm=build_dm(cfg); batch=next(iter(dm.test_dataloader())); target,x,y=batch
        for variant in ['dwt_concat','dwt_static_gate','dwt_dynamic_gate','wtam_full']:
            m=definitive.make_model(cfg,dm,variant)
            out=m(x,y,x_recon_target=target)
            assert torch.isfinite(out['loss']); out['loss'].backward()
    result={'context':a.context,'imports':imported,'config_checks':checked,'region_checks':region_checks,'day_counts':days}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'context':a.context,'imports':len(imported),'configs':len(checked),'regions':len(region_checks),'day_counts':days}))

if __name__=='__main__': main()
