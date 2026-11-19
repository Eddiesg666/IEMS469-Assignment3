"""Deterministic publication figures from the actual experiment histories."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import load

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig)
def style(ax):ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
def make(root,out):
    root=Path(root);out=Path(out);p=out/'figures';p.mkdir(parents=True,exist_ok=True);m=load(root/'manifest.json');h=np.array([c['histogram'] for c in m['clients']]); sizes=h.sum(1);ids=np.argsort(sizes)[[0,24,49,74,-1]];total=h.sum(0)
    fig,ax=plt.subplots(figsize=(9,3.2));ax.bar(np.arange(62),total,color='#306c98');ax.set(xlabel='Class index (0–61)',ylabel='Images',title='Development dataset: overall class distribution');style(ax);save(fig,p,'class_counts.png')
    fig,ax=plt.subplots(figsize=(9,3.3));a=np.vstack([total/total.sum(),h[ids]/sizes[ids,None]])*100;im=ax.imshow(a,aspect='auto',cmap='Blues');ax.set_yticks(range(6),['All']+[f'Client {i:02d} (n={sizes[i]})' for i in ids]);ax.set(xlabel='Class index',title='Five writers at sample-count quantiles: class proportions');fig.colorbar(im,ax=ax,label='Within-writer class percentage');save(fig,p,'writer_distribution.png')
    s=load(out/'selection.json'); names=[d['name'] for d in s['grid']];colors=['#247ba0','#d47d27','#2b8865','#8b589e']
    def panel(names,filename,title,metrics=('train_acc','val_acc','train_loss','val_loss'),labels=None):
        fig,axs=plt.subplots(1 if len(metrics)==2 else 2,2,figsize=(10,3.3 if len(metrics)==2 else 6.3));axs=np.ravel(axs)
        for i,name in enumerate(names):
            rows=load(out/name/'history.json');r=[j['round'] for j in rows];cfg=load(out/name/'config.json');label=labels[i] if labels else f"C={cfg['k']}%, E={cfg['epochs']}"
            for ax,key in zip(axs,metrics):ax.plot(r,[j[key]*(100 if key.endswith('acc') else 1) for j in rows],label=label,lw=1.5);ax.set(title=key.replace('_',' ').title(),xlabel='Communication round',ylabel='Accuracy (%)' if key.endswith('acc') else 'Cross-entropy');style(ax)
        axs[0].legend(fontsize=8,loc='best');fig.suptitle(title,fontsize=13);save(fig,p,filename)
    panel(names,'grid.png','Sequential FedAvg hyperparameter comparison')
    if (out/'noise_selection.json').exists():
        d=load(out/'noise_selection.json');ns=d['runs']; nn=[v['name'] for v in ns]; labs=[f"b={v['b']:g}" for v in ns];panel(nn,'noise_curves.png','Laplace noise: training on perturbed images, clean validation',('train_acc','val_acc'),labs)
        test=load(out/'serial_test.json');b=[v['b'] for v in ns];fig,ax=plt.subplots(figsize=(8,3.7));ax.plot(b,[100*v['final']['train_acc'] for v in ns],'o-',label='Train (perturbed)');ax.plot(b,[100*v['final']['clean_train_acc'] for v in ns],'s--',label='Train (clean diagnostic)');ax.plot(b,[100*test[v['name']]['accuracy'] for v in ns],'o-',label='Held-out test (clean)');ax.plot(b,[100*v['final']['val_acc'] for v in ns],'d:',label='Validation (clean)');ax.set(xlabel='Laplace scale b (normalized pixels)',ylabel='Accuracy (%)',title='Privacy–utility experiment: final accuracy vs. noise scale');ax.legend(fontsize=8);style(ax);save(fig,p,'noise_tradeoff.png')
    rows=load(out/s['selected']/'history.json');r=np.array([j['round'] for j in rows]);fig,axs=plt.subplots(1,2,figsize=(10,3.5));ax=axs[0];ax.fill_between(r,[100*j['val_client_p10'] for j in rows],[100*j['val_client_p90'] for j in rows],alpha=.22,color='#297ea8',label='10th–90th percentile');ax.plot(r,[100*j['val_client_median'] for j in rows],label='Median writer');ax.plot(r,[100*j['val_acc'] for j in rows],label='Sample-weighted global');ax.set(xlabel='Round',ylabel='Validation accuracy (%)',title='Nonstandard: writer accuracy disparity');ax.legend(fontsize=7);style(ax)
    ax=axs[1];ax.plot(r[1:],[j['drift'] for j in rows[1:]],label='Mean client update norm');ax.plot(r[1:],[j['global_step'] for j in rows[1:]],label='Global update norm');ax.set(xlabel='Round',ylabel='L2 parameter movement',title='Nonstandard: client drift and cancellation');ax.legend(fontsize=7);style(ax);save(fig,p,'diagnostics.png')
    v=load(out/'validation.json');fig,axs=plt.subplots(1,2,figsize=(10,3.5));ax=axs[0];ax.bar(range(100),v['sampling_counts'],color='#306c98');ax.axhline(v['expected_count'],color='#c44e52',ls='--',label='Expected 100');ax.set(xlabel='Client index',ylabel='Selections (1,000 rounds)',title='Uniform sampling check');ax.legend(fontsize=8);style(ax);ax=axs[1];bs=[j['b'] for j in v['noise']];ax.plot(bs,[j['expected_variance'] for j in v['noise']],label='Theory: 2b²');ax.scatter(bs,[j['variance'] for j in v['noise']],marker='x',s=55,label='Measured variance');ax.set(xlabel='Laplace scale',ylabel='Variance',title='Noise distribution calibration');ax.legend(fontsize=8);style(ax);save(fig,p,'validation.png')
    if (out/'ray_test.json').exists():
        d=load(out/'ray_test.json');panel([d['serial_name'],'ray_c4'],'ray.png','Four-client matched experiment',labels=['Sequential','Ray actors'])
    return ids.tolist()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',default='prepared');a.add_argument('--out',default='runs');q=a.parse_args();make(q.data,q.out)
