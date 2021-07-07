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

data1=np.loadtxt("/data02/tguo/batch_effect/kidney_mat.txt",dtype=np.float64)
data2=np.loadtxt("/data02/tguo/batch_effect/liver_mat.txt",dtype=np.float64)
data3=np.loadtxt("/data02/tguo/batch_effect/eso_mat.txt",dtype=np.float64)
data4=np.loadtxt("/data02/tguo/batch_effect/lung_mat.txt",dtype=np.float64)
data5=np.loadtxt("/data02/tguo/batch_effect/spleen_mat.txt",dtype=np.float64)
data6=np.loadtxt("/data02/tguo/batch_effect/pbmc1_mat.txt",dtype=np.float64)
data7=np.loadtxt("/data02/tguo/batch_effect/pbmc2_mat.txt",dtype=np.float64)
data8=np.loadtxt("/data02/tguo/batch_effect/pan_mat.txt",dtype=np.float64)

celltype1=pd.read_csv("/data02/tguo/batch_effect/kidney_celltype.txt",sep=",",header=None).values[:,0]
celltype2=pd.read_csv("/data02/tguo/batch_effect/liver_celltype.txt",sep=",",header=None).values[:,0]
celltype3=pd.read_csv("/data02/tguo/batch_effect/eso_celltype.txt",sep=",",header=None).values[:,0]
celltype4=pd.read_csv("/data02/tguo/batch_effect/lung_celltype.txt",sep=",",header=None).values[:,0]
celltype5=pd.read_csv("/data02/tguo/batch_effect/spleen_celltype.txt",sep=",",header=None).values[:,0]
celltype6=pd.read_csv("/data02/tguo/batch_effect/pbmc1_celltype.txt",sep=",",header=None).values[:,0]
celltype7=pd.read_csv("/data02/tguo/batch_effect/pbmc2_celltype.txt",sep=",",header=None).values[:,0]
celltype8=pd.read_csv("/data02/tguo/batch_effect/pan_celltype.txt",sep=",",header=None).values[:,0]

data=np.hstack((data1,data2,data3,data4,data5,data6,data7,data8))
celltype=np.hstack((celltype1,celltype2,celltype3,celltype4,celltype5,celltype6,celltype7,celltype8))
nl=np.array([len(celltype1),len(celltype2),len(celltype3),len(celltype4),len(celltype5),len(celltype6),len(celltype7),len(celltype8)])
nl1=np.array([0,len(celltype1),len(celltype2),len(celltype3),len(celltype4),len(celltype5),len(celltype6),len(celltype7),len(celltype8)])
cnl=np.cumsum(nl1)
N=8
batch=np.zeros((len(celltype),N))
for i in np.arange(N):
    batch[cnl[i]:cnl[i+1],i]=1

num=np.min(nl)
k=1
IDX=[]
for i in np.arange(N):
    idx=np.where(batch[:,i]==1)[0]
    if len(idx)>k*num:
        random.shuffle(idx)
        idx=idx[np.arange(k*num)]
    idx=idx.tolist()
    IDX=IDX+idx
IDX=np.array(IDX)

data_train=data[:,IDX]
celltype_train=celltype[IDX]
batch_train=batch[IDX,:]



# ############
data_1=deepcopy(data)
data=np.transpose(data)
data=minmax_scale(data,axis=1)
data_train=np.transpose(data_train)
data_train=minmax_scale(data_train,axis=1)

epoch=500
latent=10
class_dim=2
batch_size=64
lr=0.001
beta=0
gamma=1
alpha=1
fg_lambda=0.1
in_dim=data.shape[1]
in_category=batch.shape[1]
n=5
Loss=np.zeros((n,epoch))
for j in np.arange(n):
    dann=DANN(in_dim,in_category,latent,class_dim,lr,beta,gamma,alpha,fg_lambda,1)
    loss=dann.train(data,batch,epoch,batch_size)
    Loss[j,:]=np.array(loss)[:epoch]
    types="data_"+str(j)+"_dann.txt"
    latent_code_mean=dann.ae1.predict(data)[0]
    np.savetxt("/data02/tguo/batch_effect/allorgan/"+str(types),latent_code_mean,fmt="%.5f")
    umap_mat=umap.UMAP().fit_transform(latent_code_mean[:,class_dim:])
    types="umap_"+str(j)+"_dann.txt"
    np.savetxt("/data02/tguo/batch_effect/allorgan/"+str(types),umap_mat,fmt="%.5f")