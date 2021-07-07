import numpy as np
import random
import matplotlib.pyplot as plt
import umap.umap_ as umap
import time
from sklearn.preprocessing import minmax_scale
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pandas import Series,DataFrame
import pandas as pd
import os
from copy import deepcopy

k=1
data=np.loadtxt("/data02/tguo/batch_effect/simulate/dataset4-2_1same_group"+str(k)+"_data.txt",dtype=np.float64)
celltype=np.loadtxt("/data02/tguo/batch_effect/simulate/dataset4-2_1same_group"+str(k)+"_celltype.txt",dtype=np.str)
batch=np.loadtxt("/data02/tguo/batch_effect/simulate/dataset4-2_1same_group"+str(k)+"_batch.txt",dtype=np.int16)
data=minmax_scale(data, axis=0)
data=np.transpose(data)
N=np.min(batch)
if N>0:
    batch=batch-N

n=5
epoch=5000
latent=10
class_dim=2
batch_size=64
lr=0.001
beta=0
gamma=1
alpha=1
fg_lambda=0.1
weight_class=1
in_dim=data.shape[1]
in_category=batch.shape[1]
# Loss=np.zeros((n,epoch))
for j in np.arange(n):
    dann=DANN(in_dim,in_category,latent,class_dim,lr,beta,gamma,alpha,fg_lambda)
    loss=dann.train(data,batch,epoch,batch_size)
#     Loss[j,:]=loss[:epoch]
    types="dataset4-2_1same_group"+str(k)+"_"+str(j)+"_dann.txt"
    np.savetxt("/data02/tguo/batch_effect/simulate/"+str(types),dann.ae1.predict(data)[0],fmt="%.5f")