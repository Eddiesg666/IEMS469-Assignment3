"""Run mandatory comparisons; choose on validation only; held-out evaluated after training."""
import argparse, os
from pathlib import Path
from data import load,dump
from fedavg import config,train,test,read_weights,RayRunner

def serial(root,out,rounds,seed):
    results=[]; out=Path(out)
    for k in (4,10):
        for e in (1,5):
            name=f'c{k}_e{e}'; cfg=config(k,e,rounds,seed)
            rows,w=train(root,out/name,cfg); results.append(dict(name=name,k=k,epochs=e,final=rows[-1]))
    selected=max(results,key=lambda d:d['final']['val_acc']); base=max([d for d in results if d['k']==10],key=lambda d:d['final']['val_acc']);
    dump(out/'selection.json',dict(grid=results,selected=selected['name'],dp_base=base['name'],dp_epochs=base['epochs'],rounds=rounds,seed=seed,dataset=load(Path(root)/'manifest.json')['fingerprint']))
    for b in (.1,.3,.7,1.5):
        train(root,out/f'noise_{b:g}',config(10,base['epochs'],rounds,seed,b=b))
    # Noise selection uses a predeclared validation-only tolerance of 3 percentage points.
    names=[base['name']]+[f'noise_{b:g}' for b in (.1,.3,.7,1.5)]
    ds=[dict(name=n,b=load(out/n/'config.json')['b'],final=load(out/n/'history.json')[-1]) for n in names]
    eligible=[d for d in ds if d['b']>0 and d['final']['val_acc']>=ds[0]['final']['val_acc']-.03]
    recommendation=max(eligible,key=lambda d:d['b']) if eligible else max(ds[1:],key=lambda d:d['final']['val_acc'])
    dump(out/'noise_selection.json',dict(runs=ds,recommended=recommendation['name'],within_tolerance=bool(eligible),tolerance=.03))
    # No test score is available until all models and choices are frozen.
    report={n:test(root,read_weights(out/n/'model_final.npz')) for n in dict.fromkeys([selected['name']]+names)}
    dump(out/'serial_test.json',report);print('SERIAL DONE',report.keys(),flush=True)

def parallel(root,out,cpus,gpus):
    out=Path(out); selection=load(out/'selection.json'); rounds=selection['rounds']; seed=selection['seed'];
    e=selection['dp_epochs']; cfg=config(4,e,rounds,seed,backend='ray'); runner=RayRunner(cpus,gpus)
    try: rows,w=train(root,out/'ray_c4',cfg,runner)
    finally: runner.close()
    baseline=out/f'c4_e{e}'; serial_w=read_weights(baseline/'model_final.npz'); from fedavg import l2
    sr=load(baseline/'history.json'); diff=l2(w,serial_w); assert all(a['selected']==b['selected'] for a,b in zip(rows,sr))
    dump(out/'ray_test.json',dict(test=test(root,w),same_schedule=True,serial_name=baseline.name,final_parameter_l2=diff,serial_val=sr[-1]['val_acc'],ray_val=rows[-1]['val_acc'],serial_update_seconds=sum(r['update_seconds'] for r in sr),ray_update_seconds=sum(r['update_seconds'] for r in rows),serial_total_seconds=sum(r['seconds'] for r in sr),ray_total_seconds=sum(r['seconds'] for r in rows),cpus=cpus,gpus=gpus))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['serial','ray']);p.add_argument('--data',default='prepared');p.add_argument('--out',default='runs');p.add_argument('--rounds',type=int,default=60);p.add_argument('--seed',type=int,default=2026);p.add_argument('--cpus',type=int,default=4);p.add_argument('--gpus',type=int,default=0);a=p.parse_args()
    if a.mode=='serial':serial(a.data,a.out,a.rounds,a.seed)
    else:parallel(a.data,a.out,a.cpus,a.gpus)
