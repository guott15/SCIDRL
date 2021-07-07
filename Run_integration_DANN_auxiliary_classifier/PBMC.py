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

types='allsame'
data=np.loadtxt("/data02/tguo/batch_effect/PBMC/9tech_count_"+str(types)+".txt",dtype=np.float64)
celltype=[]
with open("/data02/tguo/batch_effect/PBMC/9tech_celltype_"+str(types)+".txt") as inputfile:
    for line in inputfile:
        line=line.strip("\n")
        celltype.append(line)
celltype=np.array(celltype)
batch=np.loadtxt("/data02/tguo/batch_effect/PBMC/9tech_batch_"+str(types)+".txt",dtype=np.int16)
L=list()
for i in range(batch.shape[1]):
    L.append(len(np.where(batch[:,i]==1)[0]))
N=np.min(np.array(L))
data_train=np.zeros((data.shape[0],1))
celltype_train=np.zeros(1)
batch_train=np.zeros((1,batch.shape[1]))
k=3
for i in range(batch.shape[1]):
    idx=np.where(batch[:,i]==1)[0]
    if len(idx)>k*N:
        num=k*N
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
fg_lambda=10
weight_class=1
in_dim=data_train.shape[1]
in_category=batch_train.shape[1]
n=5
Loss=np.zeros((n,epoch))
for j in np.arange(1,n):
    expr=np.hstack((data_train,np.zeros((data_train.shape[0],latent))))
    dann=DANN(in_dim,in_category,latent,class_dim,lr,beta,gamma,alpha,fg_lambda,weight_class)
    loss=dann.train(expr,batch_train,epoch,batch_size)
    expr=np.hstack((data,np.zeros((data.shape[0],latent))))
    latent_code_mean=dann.ae1.predict(expr)[0]
    types="9tech_allsame_count_"+str(j)+"_dann.txt"
    np.savetxt("/data02/tguo/batch_effect/PBMC/"+str(types),latent_code_mean,fmt="%.5f")
    a=np.hstack((np.zeros((latent_code_mean.shape[0],2)),latent_code_mean[:,2:]))
    expr=np.hstack((data,a))
    correct=dann.ae1.predict(expr)[1]
    types="9tech_allsame_exp_"+str(j)+"_dann.txt"
    np.savetxt("/data02/tguo/batch_effect/PBMC/"+str(types),correct,fmt="%.5f")