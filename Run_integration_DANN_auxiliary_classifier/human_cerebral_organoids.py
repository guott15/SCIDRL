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

data=np.loadtxt("/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/mat.txt",dtype=np.float64)
celltype=[]
with open("/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/celltype.txt") as inputfile:
    for line in inputfile:
        line=line.strip("\n")
        celltype.append(line)
celltype=np.array(celltype)
batch=np.loadtxt("/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/batch.txt",dtype=np.int32)
idx=np.where(celltype=="Unknown")[0]

batch1=deepcopy(batch)-1
N=len(np.unique(batch1))
batch=np.zeros((len(batch1),N))
nl=np.zeros(N)
for i in np.arange(N):
    batch[np.where(batch1==i)[0],i]=1
    nl[i]=len(np.where(batch1==i)[0])
nl=np.int16(nl)
mnl=3000

data_train=np.zeros((data.shape[0],1))
celltype_train=np.zeros(1)
batch_train=np.zeros((1,batch.shape[1]))
k=1
for i in range(batch.shape[1]):
    idx=np.where(batch[:,i]==1)[0]
    if len(idx)>k*mnl:
        num=k*mnl
    else:
        num=len(idx)
    random.shuffle(idx)
    data_train=np.hstack((data_train,data[:,idx[np.arange(num)]]))
    celltype_train=np.hstack((celltype_train,celltype[idx[np.arange(num)]]))
    batch_train=np.vstack((batch_train,batch[idx[np.arange(num)],:]))

data=np.transpose(data)
data_1=deepcopy(data)
data=minmax_scale(data,axis=1)
data_train=np.transpose(data_train)
data_train=minmax_scale(data_train,axis=1)
data_train=data_train[1:data_train.shape[0],:]
celltype_train=celltype_train[1:len(celltype_train)]
batch_train=batch_train[1:batch_train.shape[0],:]

epoch=500
latent=10
class_dim=2
batch_size=64
lr=0.001
beta=0
gamma=1
alpha=1
fg_lambda=0.1
weight_class=1
in_dim=data_train.shape[1]
in_category=batch_train.shape[1]
n=5
# Loss=np.zeros((n,epoch))
for j in np.arange(0,n):
    dann=DANN(in_dim,in_category,latent,class_dim,lr,beta,gamma,alpha,fg_lambda,weight_class)
    loss=dann.train(data_train,batch_train,epoch,batch_size)
#     Loss[j,:]=np.array(loss)[:epoch]
    types="count_"+str(j)+"_dann.txt"
    np.savetxt("/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/"+str(types),dann.ae1.predict(data)[0],fmt="%.5f")