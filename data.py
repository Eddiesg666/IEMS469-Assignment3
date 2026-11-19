"""Prepare writer-preserving EMNIST partitions. No test labels used in model selection."""
import argparse, hashlib, json, math, shutil, tempfile, zipfile
from pathlib import Path
import numpy as np


def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); tmp.replace(path)


def load(path): return json.loads(Path(path).read_text())


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8], 'little') % (2**32)


def split_indices(n, seed):
    if n < 2: raise ValueError('Each writer needs at least two examples for nonempty train and validation sets.')
    p=np.random.default_rng(seed).permutation(n)
    k=max(1,min(n-1,int(math.floor(.8*n))))
    return p[:k],p[k:]


def arrays(x,y):
    x=np.asarray(x,dtype=np.float32); y0=np.asarray(y).reshape(-1)
    if not np.issubdtype(y0.dtype,np.number) or not np.isfinite(y0).all() or np.any(y0!=np.floor(y0)):
        raise ValueError('Labels must be integer class indices 0..61; inspect dataset mapping.')
    y=y0.astype(np.int64)
    if len(y)==0 or x.size!=len(y)*784 or not np.isfinite(x).all() or np.any(y<0) or np.any(y>=62):
        raise ValueError(f'Invalid EMNIST image or label shape: {x.shape}, {y.shape}')
    return x.reshape(-1,784),y


def read_one(path):
    path=Path(path); ext=path.suffix.lower()
    if ext=='.json':
        obj=load(path); d=obj.get('user_data',obj)
        if 'x' in d and 'y' in d: yield 'global', arrays(d['x'],d['y']); return
        if not isinstance(d,dict): raise ValueError('Expected LEAF user_data or per-writer mapping')
        for u,v in d.items():
            if not isinstance(v,dict) or 'x' not in v or 'y' not in v: raise ValueError(f'Unsupported JSON layout: {path}')
            yield str(u),arrays(v['x'],v['y'])
    elif ext in ('.h5','.hdf5'):
        import h5py
        with h5py.File(path,'r') as f:
            g=f['examples'] if 'examples' in f else f
            for u in g:
                v=g[u]; xk=next((k for k in ('pixels','x','images') if k in v),None); yk=next((k for k in ('label','labels','y') if k in v),None)
                if xk is None or yk is None: raise ValueError(f'Unsupported HDF5 group {u}')
                yield str(u),arrays(v[xk][...],v[yk][...])
    elif ext=='.npy':
        # Canvas archive deliberately uses pickled object dictionaries; only load trusted course data.
        d=np.load(path,allow_pickle=True)
        if d.ndim!=1 or d.dtype!=object: raise ValueError('Expected Canvas object array')
        for i,v in enumerate(d):
            if not isinstance(v,dict) or not {'images','labels'}<=set(v): raise ValueError('Unexpected Canvas record')
            yield f'{i:03d}',arrays(v['images'],v['labels'])
    elif ext=='.npz':
        with np.load(path,allow_pickle=False) as d:
            xk=next((k for k in ('x','images','pixels') if k in d),None); yk=next((k for k in ('y','labels','label') if k in d),None)
            if xk is None or yk is None: raise ValueError('NPZ requires x/y or images/labels')
            x,y=arrays(d[xk],d[yk]); key=next((k for k in ('client_ids','user_ids','writers','users') if k in d),None)
            if key is None: yield 'global',(x,y)
            else:
                ids=d[key].astype(str).reshape(-1)
                if len(ids)!=len(y): raise ValueError('NPZ requires one writer id per example')
                for u in np.unique(ids): yield str(u),(x[ids==u],y[ids==u])
    else: raise ValueError(f'Unsupported source {path}')

EXT={'.json','.h5','.hdf5','.npz','.npy'}

def files(path):
    p=Path(path)
    if p.is_file() and p.suffix.lower() in EXT: return [p]
    if p.is_dir(): return sorted(q for q in p.rglob('*') if q.is_file() and q.suffix.lower() in EXT and '__MACOSX' not in q.parts and not q.name.startswith('.'))
    raise ValueError(f'Expected extracted data directory or supported file: {p}')


def discover(root):
    root=Path(root); fs=files(root)
    tr=[];te=[]
    for p in fs:
        rel=str(p.relative_to(root)).lower() if root.is_dir() else p.name.lower()
        if 'test' in rel and 'train' not in rel: te.append(p)
        elif 'train' in rel and 'test' not in rel: tr.append(p)
    if not tr or not te: raise ValueError('Could not unambiguously identify train/test. Specify --train and --test separately; inspect archive layout.')
    return tr,te


def read_many(paths):
    grouped={}
    for p in paths:
        for u,(x,y) in read_one(p): grouped.setdefault(u,[]).append((x,y))
    return {u:(np.concatenate([v[0] for v in a]),np.concatenate([v[1] for v in a])) for u,a in grouped.items()}


def prepare(train_files,test_files,out,seed=2026,synthetic=False):
    out=Path(out)
    if out.exists(): raise FileExistsError(f'{out} already exists; use a new path, not an overwrite.')
    if set(map(lambda p:Path(p).resolve(),train_files)) & set(map(lambda p:Path(p).resolve(),test_files)): raise ValueError('Train and test sources overlap')
    tr=read_many(train_files)
    if not synthetic and len(tr)!=100: raise ValueError(f'Expected exactly 100 training writers, found {len(tr)}. Do not invent writer IDs; inspect Canvas files.')
    if len(tr)<10: raise ValueError('Need at least 10 writers for experimental configurations')
    lo=min(float(x.min()) for x,y in tr.values()); hi=max(float(x.max()) for x,y in tr.values())
    if lo<0 or hi>255: raise ValueError(f'Expected pixels in [0,1] or [0,255], found [{lo},{hi}]')
    # Single convention inferred from training only; fixed for validation and held-out test.
    scale=255.0 if hi>1.00001 else 1.0
    def norm(x):
        if x.min()<0 or x.max()>scale+1e-5: raise ValueError('Pixel range inconsistent with training convention')
        return (x/scale).astype(np.float32)
    out.parent.mkdir(parents=True,exist_ok=True); tmp=Path(tempfile.mkdtemp(prefix=out.name+'.',dir=out.parent))
    try:
        (tmp/'clients').mkdir(); meta=[]; digest=hashlib.sha256()
        for i,(u,(x,y)) in enumerate(sorted(tr.items())):
            x=norm(x); a,b=split_indices(len(y),stable_seed(seed,u,'split'))
            name=f'client_{i:03d}.npz'; np.savez_compressed(tmp/'clients'/name, x_train=x[a],y_train=y[a],x_val=x[b],y_val=y[b],train_idx=a,val_idx=b)
            hist=np.bincount(y,minlength=62).tolist()
            digest.update(u.encode()); digest.update(x.tobytes()); digest.update(y.tobytes())
            meta.append(dict(id=i,file=name, n_train=len(a),n_val=len(b), histogram=hist))
        # Keep test separate; no test metrics or distributions are exposed to the training routine.
        te=read_many(test_files); x=np.concatenate([v[0] for v in te.values()]); y=np.concatenate([v[1] for v in te.values()]); x=norm(x)
        np.savez_compressed(tmp/'heldout.npz',x=x,y=y)
        digest.update(x.tobytes()); digest.update(y.tobytes())
        m=dict(seed=seed,synthetic=synthetic, fingerprint=digest.hexdigest(),pixel_divisor=scale, clients=meta, test_n=len(y), test_writer_count=len(te), source_train=[str(Path(p).name) for p in train_files],source_test=[str(Path(p).name) for p in test_files])
        dump(tmp/'manifest.json',m); tmp.rename(out)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)
    return m


def inspect(path):
    p=Path(path)
    if p.is_file() and zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            print('\n'.join(f'{i.filename} ({i.file_size:,} bytes)' for i in z.infolist()[:100])); print('Total entries:',len(z.infolist()))
    else:
        for q in sorted(p.rglob('*'))[:100] if p.is_dir() else [p]: print(q)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--inspect'); p.add_argument('--data'); p.add_argument('--train'); p.add_argument('--test'); p.add_argument('--out',default='prepared');p.add_argument('--seed',type=int,default=2026);a=p.parse_args()
    if a.inspect: inspect(a.inspect); return
    if a.train and a.test: tr,te=files(a.train),files(a.test)
    elif a.data: tr,te=discover(a.data)
    else: p.error('Provide --data or both --train and --test')
    m=prepare(tr,te,a.out,a.seed)
    print(f"Prepared {len(m['clients'])} clients; train={sum(c['n_train'] for c in m['clients'])}, validation={sum(c['n_val'] for c in m['clients'])}, held-out={m['test_n']}; pixel divisor={m['pixel_divisor']}")

if __name__=='__main__': main()
