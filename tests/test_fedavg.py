import numpy as np
import pytest
import torch
from data import split_indices,arrays,stable_seed
from fedavg import aggregate, perturb,model,init,optimize,l2,sample_clients,config

def test_model(): assert model()(torch.zeros(2,784)).shape==(2,62)
@pytest.mark.parametrize('n',[2,5,13,344])
def test_split(n):
    a,b=split_indices(n,4); assert len(a)+len(b)==n and not set(a)&set(b) and len(a)>0 and len(b)>0
    assert np.array_equal(a,split_indices(n,4)[0])
def test_singleton():
    with pytest.raises(ValueError):split_indices(1,2)
def test_weighted(): assert aggregate([({'v':np.array([1.])},1),({'v':np.array([9.])},3)])['v'][0]==7
@pytest.mark.parametrize('updates',[[],[({'v':np.array([1.])},0)],[({'v':np.array([np.nan])},1)]])
def test_invalid_update(updates):
    with pytest.raises(ValueError):aggregate(updates)
def test_sampling():
    a=sample_clients(100,10,2026,1);assert len(set(a))==10 and a==sample_clients(100,10,2026,1)
    with pytest.raises(ValueError):sample_clients(100,101,1,1)
def test_noise():
    x=np.zeros(10000,dtype=np.float32);a=perturb(x,.3,10);assert np.array_equal(a,perturb(x,.3,10)) and np.array_equal(x,perturb(x,0,1)) and abs(a.var()-.18)<.02
    with pytest.raises(ValueError):perturb(x,-1,1)
def test_invalid_label():
    with pytest.raises(ValueError):arrays(np.zeros((1,28,28)),[62])
def test_single_label_training():
    torch.set_num_threads(1);w=init(1);x=np.ones((8,784),dtype=np.float32);y=np.ones(8,dtype=np.int64);v=optimize(w,x,y,2,.1,4,1);assert l2(w,v)>0
