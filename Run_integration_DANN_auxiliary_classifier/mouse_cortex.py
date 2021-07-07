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

types="allsame"
data=np.loadtxt("/data02/tguo/batch_effect/mouse_cortex/"+str(types)+"_mat.txt",dtype=np.float64)
celltype=pd.read_csv("/data02/tguo/batch_effect/mouse_cortex/"+str(types)+"_celltype.txt",sep=",",header=None).values[:,0]
batch=np.loadtxt("/data02/tguo/batch_effect/mouse_cortex/"+str(types)+"_batch.txt",dtype=np.int16)

n=min(len(np.where(batch[:,0]==1)[0]),len(np.where(batch[:,1]==1)[0]),len(np.where(batch[:,2]==1)[0]),len(np.where(batch[:,3]==1)[0]))
idx=np.arange(len(batch))
random.shuffle(idx)
train_batch=batch[idx,:]
train_celltype=celltype[idx]
train_data=data[:,idx]
k=1
IDX=[]
for i in np.arange(4):
    if len(np.where(train_batch[:,i]==1)[0])>k*n:
        idx=np.where(train_batch[:,i]==1)[0][range(k*n)]
    else:
        idx=np.where(train_batch[:,i]==1)[0]
    IDX=IDX+idx.tolist()

train_batch=train_batch[IDX,:]
train_celltype=train_celltype[IDX]
train_data=train_data[:,IDX]

# # ############
data_1=deepcopy(np.transpose(data))
data=np.transpose(minmax_scale(data,axis=0))
train_data=np.transpose(minmax_scale(train_data,axis=0))

epoch=500
latent=10
class_dim=2
batch_size=64
lr=0.001
beta=0
gamma=1
alpha=1
fg_lambda=10
weight_class=1
in_dim=train_data.shape[1]
in_category=train_batch.shape[1]
n=5
Loss=np.zeros((n,epoch))

for j in np.arange(2,n):
    expr=np.hstack((train_data,np.zeros((train_data.shape[0],latent))))
    dann=DANN(in_dim,in_category,latent,class_dim,lr,beta,gamma,alpha,fg_lambda,weight_class)
    loss=dann.train(expr,train_batch,epoch,batch_size)
#     Loss[j,:]=np.array(loss)[:epoch]
    expr=np.hstack((data,np.zeros((data.shape[0],latent))))
    latent_code_mean=dann.ae1.predict(expr)[0]
    types="allsame_count_"+str(j)+"_dann.txt"
    np.savetxt("/data02/tguo/batch_effect/mouse_cortex/"+str(types),latent_code_mean,fmt="%.5f")
    a=np.hstack((np.zeros((latent_code_mean.shape[0],2)),latent_code_mean[:,2:]))
    expr=np.hstack((data,a))
    correct=dann.ae1.predict(expr)[1]
    types="allsame_exp_"+str(j)+"_dann.txt"
    np.savetxt("/data02/tguo/batch_effect/mouse_cortex/"+str(types),correct,fmt="%.5f")