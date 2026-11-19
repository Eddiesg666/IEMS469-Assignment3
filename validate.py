"""Quantitative algorithm checks on actual local partitions; never reads held-out examples."""
from pathlib import Path
import argparse, math
import numpy as np
import torch
from fedavg import (aggregate,init,model,set_state,state,optimize,client_data,l2,perturb,sample_clients,local,config,metrics)
from data import load,dump,stable_seed

def validate(root,out):
    torch.set_num_threads(1); root=Path(root); meta=load(root/'manifest.json'); cs=meta['clients']; w=init(2026)
    counts=[]; valid=True; label_missing=0; distributions=[]
    for c in cs:
        with np.load(root/'clients'/c['file']) as d:
            a,b=d['train_idx'],d['val_idx']; n=len(a)+len(b); valid &= bool(len(np.intersect1d(a,b))==0 and len(np.unique(np.r_[a,b]))==n and len(a)==int(.8*n)); counts.append((len(a),len(b)))
        h=np.asarray(c['histogram']); distributions.append(h);label_missing+=int((h==0).sum())
    h=np.asarray(distributions); totals=h.sum(0); p=h/h.sum(1,keepdims=True); global_p=totals/totals.sum(); tv=.5*np.abs(p-global_p).sum(1)
    # choose four writers with unequal sizes, taking one full-batch SGD step each.
    pick=list(dict.fromkeys([int(np.argmin(h.sum(1))),int(np.argmax(h.sum(1))),25,75,50,0])); chosen=[cs[i] for i in pick[:4]]; xy=[client_data(root,c,0.,2026)[:2] for c in chosen]
    lr=.08; local_w=[optimize(w,x,y,1,lr,len(y),2026) for x,y in xy]; ns=[len(y) for x,y in xy]
    wa=aggregate(list(zip(local_w,ns))); wu=aggregate([(v,1) for v in local_w]);
    # Independent reference calculates weighted gradient without sharing/concatenating raw data.
    grads=[]
    for x,y in xy:
        m=set_state(model(),w); loss=torch.nn.functional.cross_entropy(m(torch.from_numpy(x)),torch.from_numpy(y));loss.backward();grads.append({k:v.grad.detach().numpy().copy() for k,v in m.named_parameters()})
    g=aggregate(list(zip(grads,ns))); reference={k:w[k]-lr*g[k] for k in w}; eq=l2(wa,reference); un=l2(wu,reference)
    # Separate exact counterexample: unequal sample weights and identical updates.
    analytic=float(aggregate([({'x':np.array([1.],dtype=np.float32)},1),({'x':np.array([9.],dtype=np.float32)},3)])['x'][0]); idem=l2(aggregate([(w,2),(w,7)]),w)
    # Check greater local epochs change drift from same initial weights and same batches.
    drift={}; loss_after={}
    for e in (1,5):
        us=[optimize(w,x,y,e,lr,32,2026) for x,y in xy];drift[str(e)]=float(np.average([l2(v,w) for v in us],weights=ns)); loss_after[str(e)]=float(sum(metrics(set_state(model(),v),x,y)[0] for v,(x,y) in zip(us,xy))/sum(ns))
    # Selection fairness across independent deterministic round seeds.
    draws=np.zeros(len(cs),int); unique=True
    for r in range(1000):
        ids=sample_clients(len(cs),10,2026,r); unique &= len(set(ids))==10;draws[ids]+=1
    expected=1000*10/len(cs); noise=[]; x=xy[0][0][:20]; a=np.zeros((1000000,),np.float32)
    for b in (0.,.1,.3,.7,1.5):
        z=perturb(a,b,19); sample=perturb(x,b,42); residual=sample-x
        noise.append(dict(b=b,mean=float(z.mean()),variance=float(z.var()),expected_variance=2*b*b,mae=float(np.abs(z).mean()),expected_mae=b,actual_image_mse=float(np.mean(residual**2)),fixed_repeat=bool(np.array_equal(sample,perturb(x,b,42))),outside_unit=float(np.mean((sample<0)|(sample>1)))))
    passed=bool(valid and unique and eq<1e-5 and un>eq and analytic==7 and idem<1e-7 and all(abs(v['variance']-v['expected_variance'])<max(1e-7,.02*v['expected_variance']) for v in noise) and all(v['fixed_repeat'] for v in noise))
    result=dict(passed=passed,split_disjoint_exact=valid,total_train=sum(a for a,b in counts),total_val=sum(b for a,b in counts),min_client=int(h.sum(1).min()),max_client=int(h.sum(1).max()),class_count=int((totals>0).sum()),min_class=int(totals.min()),max_class=int(totals.max()),missing_client_class_pairs=label_missing,median_client_tv=float(np.median(tv)),client_tv=tv.tolist(),selected_gradient_clients=[c['id'] for c in chosen],gradient_n=ns,weighted_reference_l2=eq,uniform_reference_l2=un,analytic_weighted=analytic,identical_update_l2=idem,drift=drift,local_loss=loss_after,sampling_counts=draws.tolist(),expected_count=expected,sampling_min=int(draws.min()),sampling_max=int(draws.max()),sampling_unique=unique,noise=noise)
    dump(Path(out)/'validation.json',result);print('Quantitative validation:',passed,'gradient error',eq,'unweighted error',un,'drift',drift,'noise scales:',len(noise),flush=True)
    if not passed:raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='prepared');p.add_argument('--out',default='runs');a=p.parse_args();validate(a.data,a.out)
