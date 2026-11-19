"""From-scratch, sample-weighted FedAvg. Ray is used only to schedule selected clients."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import csv, math, time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data import dump,load,stable_seed


def model(): return nn.Sequential(nn.Linear(784,128),nn.ReLU(),nn.Linear(128,62))

def state(m): return {k:v.detach().cpu().numpy().copy() for k,v in m.state_dict().items()}

def set_state(m,w): m.load_state_dict({k:torch.from_numpy(np.array(v,copy=True)) for k,v in w.items()}); return m

def init(seed): torch.manual_seed(seed); return state(model())

def save_weights(path,w):
    p=Path(path); tmp=p.with_suffix('.tmp.npz'); np.savez(tmp,**w); tmp.replace(p)

def read_weights(path):
    with np.load(path,allow_pickle=False) as d: return {k:d[k] for k in d.files}

def aggregate(updates):
    if not updates or any(n<=0 for w,n in updates): raise ValueError('Need nonempty positive-weight client updates')
    total=sum(n for _,n in updates); keys=list(updates[0][0]); out={}
    for k in keys:
        shape=updates[0][0][k].shape
        if any(set(w)!=set(keys) or w[k].shape!=shape or not np.isfinite(w[k]).all() for w,n in updates): raise ValueError('Invalid client parameter update')
        out[k]=sum(w[k].astype(np.float64)*(n/total) for w,n in updates).astype(np.float32)
    return out

def sample_clients(n,k,seed,r):
    if not 1<=k<=n: raise ValueError('Invalid number of clients')
    return np.random.default_rng(stable_seed(seed,'round',r)).choice(n,k,replace=False).tolist()

def perturb(x,b,seed):
    if not math.isfinite(b) or b<0: raise ValueError('Laplace scale must be finite and nonnegative')
    if b==0: return x.copy()
    return (x+np.random.default_rng(seed).laplace(0,b,x.shape).astype(np.float32)).astype(np.float32)

@lru_cache(maxsize=100)
def _client_data(root,file,cid,b,seed):
    with np.load(Path(root)/'clients'/file,allow_pickle=False) as d:
        x=1.0-d['x_train']; y=d['y_train']; xv=1.0-d['x_val']; yv=d['y_val']
    return x,y,xv,yv,perturb(x,b,stable_seed(seed,'noise',cid))

def client_data(root,c,b,seed):
    return _client_data(str(Path(root).resolve()),c['file'],c['id'],b,seed)

def optimize(w,x,y,epochs,lr,batch,seed,device='cpu'):
    if len(y)==0 or epochs<1 or batch<1 or lr<=0: raise ValueError('Invalid client optimization configuration')
    m=set_state(model(),w).to(device); m.train(); opt=torch.optim.SGD(m.parameters(),lr=lr)
    xx=torch.from_numpy(x).to(device); yy=torch.from_numpy(y).long().to(device); rng=np.random.default_rng(seed)
    for _ in range(epochs):
        for i in _batches(rng.permutation(len(y)),batch):
            opt.zero_grad(); loss=nn.functional.cross_entropy(m(xx[i]),yy[i]); loss.backward(); opt.step()
    return state(m)

def _batches(idx,batch):
    for j in range(0,len(idx),batch): yield idx[j:j+batch]

def local(root,c,w,cfg,r,device='cpu'):
    t=time.perf_counter(); x,y,xv,yv,z=client_data(root,c,cfg['b'],cfg['seed'])
    update=optimize(w,z,y,cfg['epochs'],cfg['lr'],cfg['batch'],stable_seed(cfg['seed'],'sgd',r,c['id']),device)
    return update,len(y),time.perf_counter()-t

def metrics(m,x,y,device='cpu'):
    m.eval(); loss=0.; correct=0
    with torch.no_grad():
        for start in range(0,len(y),512):
            xx=torch.from_numpy(x[start:start+512]).to(device); yy=torch.from_numpy(y[start:start+512]).long().to(device); z=m(xx)
            loss+=float(nn.functional.cross_entropy(z,yy,reduction='sum').cpu()); correct+=int((z.argmax(1)==yy).sum().cpu())
    return loss,correct,len(y)

def global_metrics(root,clients,w,cfg):
    # Evaluation is logically local: each partition returns only loss sums, correct counts and sample counts.
    m=set_state(model(),w); sums={k:np.zeros(3) for k in ('train','clean_train','val')}; per=[]
    for c in clients:
        x,y,xv,yv,z=client_data(root,c,cfg['b'],cfg['seed']); tr=metrics(m,z,y); cl=tr if cfg['b']==0 else metrics(m,x,y); va=metrics(m,xv,yv)
        for k,v in [('train',tr),('clean_train',cl),('val',va)]: sums[k]+=v
        per.append(va[1]/va[2])
    out={}
    for k,(loss,correct,n) in sums.items(): out.update({k+'_loss':float(loss/n),k+'_acc':float(correct/n)})
    out['val_client_p10']=float(np.quantile(per,.1)); out['val_client_median']=float(np.median(per)); out['val_client_p90']=float(np.quantile(per,.9))
    return out

def l2(w,v): return math.sqrt(sum(float(np.sum((w[k].astype(np.float64)-v[k])**2)) for k in w))

class ClientActor:
    def __init__(self,root,c,device):
        torch.set_num_threads(1); self.root=root; self.c=c; self.device=device
    def update(self,w,cfg,r): return local(self.root,self.c,w,cfg,r,self.device)

class RayRunner:
    def __init__(self,cpus=4,gpus=0):
        import ray
        self.ray=ray; self.gpus=gpus
        if cpus<1 or gpus<0: raise ValueError('Invalid Ray resources')
        if not ray.is_initialized(): ray.init(num_cpus=cpus,num_gpus=gpus,include_dashboard=False,log_to_driver=False)
        resources=ray.cluster_resources(); self.width=min(4,int(resources.get('CPU',0)),int(resources.get('GPU',0)) if gpus else 4)
        if self.width<1: raise RuntimeError('Ray has no eligible CPU/GPU slot')
        self.actor=ray.remote(num_cpus=1,num_gpus=1 if gpus else 0)(ClientActor)
    def run(self,root,clients,w,cfg,r):
        ray=self.ray; result=[]; ref=ray.put(w)
        for i in range(0,len(clients),self.width):
            actors=[]
            try:
                for c in clients[i:i+self.width]: actors.append(self.actor.remote(str(Path(root).resolve()),c,'cuda' if self.gpus else 'cpu'))
                result.extend(ray.get([a.update.remote(ref,cfg,r) for a in actors]))
            finally:
                for a in actors: ray.kill(a,no_restart=True)
        return result
    def close(self): self.ray.shutdown()

def config(k,epochs,rounds,seed=2026,b=0.,lr=.08,batch=32,backend='serial'):
    if k<1 or epochs<1 or rounds<1 or lr<=0 or batch<1 or b<0 or not math.isfinite(b): raise ValueError('Invalid config')
    return dict(k=k,epochs=epochs,rounds=rounds,seed=seed,b=b,lr=lr,batch=batch,backend=backend)

def train(root,out,cfg,runner=None):
    _client_data.cache_clear()
    torch.set_num_threads(1)
    root=Path(root); out=Path(out); out.mkdir(parents=True,exist_ok=True); meta=load(root/'manifest.json'); clients=meta['clients']; n=len(clients)
    if cfg['k']>max(1,int(n*.1)) and not meta['synthetic']: raise ValueError('Client fraction exceeds 10%')
    if cfg['backend']=='ray' and (runner is None or cfg['k']!=4): raise ValueError('Ray run requires a runner and exactly 4 clients')
    tag=dict(cfg); tag['dataset']=meta['fingerprint']; stamp=out/'config.json'; records=out/'history.json'
    if stamp.exists():
        if load(stamp)!=tag: raise ValueError(f'Configuration mismatch in {out}; choose a new results directory')
    else: dump(stamp,tag)
    if records.exists() and (out/'model_final.npz').exists():
        existing=load(records)
        if len(existing)==cfg['rounds']+1 and [j['round'] for j in existing]==list(range(cfg['rounds']+1)):
            return existing,read_weights(out/'model_final.npz')
    rows=[]; w=init(cfg['seed'])
    # Each round is committed atomically via a separate file; incomplete rounds are safely redone.
    for r in range(0,cfg['rounds']+1):
        q=out/f'round_{r:04d}.npz'; j=out/f'round_{r:04d}.json'
        if j.exists() and q.exists(): rows.append(load(j)); w=read_weights(q); continue
        if r==0:
            row=dict(round=0,selected=[],seconds=0.,local_seconds_sum=0.,update_seconds=0.,drift=0.,global_step=0.,cancellation=0.,**global_metrics(root,clients,w,cfg))
        else:
            selected=sample_clients(n,cfg['k'],cfg['seed'],r); chosen=[clients[i] for i in selected]; start=time.perf_counter()
            res=runner.run(root,chosen,w,cfg,r) if runner else [local(root,c,w,cfg,r) for c in chosen]
            update_time=time.perf_counter()-start; next_w=aggregate([(v,count) for v,count,t in res]); drift=sum(count*l2(v,w) for v,count,t in res)/sum(count for v,count,t in res); step=l2(next_w,w)
            row=dict(round=r,selected=selected,seconds=0.,local_seconds_sum=sum(t for v,count,t in res),update_seconds=update_time,drift=drift,global_step=step,cancellation=step/drift if drift else 0.,**global_metrics(root,clients,next_w,cfg)); row['seconds']=time.perf_counter()-start; w=next_w
        save_weights(q,w); dump(j,row); rows.append(row); dump(records,rows)
        if r==0 or r%5==0 or r==cfg['rounds']: print(f"{out.name} round {r}/{cfg['rounds']} train={row['train_acc']:.3%} clean_train={row['clean_train_acc']:.3%} val={row['val_acc']:.3%} ({row['seconds']:.2f}s)",flush=True)
    dump(records,rows); save_weights(out/'model_final.npz',w)
    with (out/'history.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    return rows,w

def test(root,w):
    with np.load(Path(root)/'heldout.npz',allow_pickle=False) as d:
        loss,correct,n=metrics(set_state(model(),w),1.0-d['x'],d['y']); hist=np.bincount(d['y'],minlength=62).tolist()
    return dict(loss=loss/n,accuracy=correct/n,n=int(n),correct=int(correct),histogram=hist)
