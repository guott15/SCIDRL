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

types='8same-1'
data=np.loadtxt("/data02/tguo/batch_effect/Pancreas/baron_muraro_"+str(types)+"_data.txt", dtype=np.float64)
celltype=np.loadtxt("/data02/tguo/batch_effect/Pancreas/baron_muraro_"+str(types)+"_celltype.txt", dtype=np.str)
batch=np.loadtxt("/data02/tguo/batch_effect/Pancreas/baron_muraro_"+str(types)+"_batch.txt", dtype=np.int16)
data=minmax_scale(data,feature_range=(0,1),axis=1)

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
in_dim=data.shape[1]
in_category=batch.shape[1]
n=5
# Loss=np.zeros((n,epoch))
for j in np.arange(n):
    expr=np.hstack((data,np.zeros((data.shape[0],latent))))
    dann=DANN(in_dim,in_category,latent,class_dim,lr,beta,gamma,alpha,fg_lambda)
    loss=dann.train(expr,batch,epoch,batch_size)
#     Loss[j,:]=np.array(loss)[:epoch]
    types1=str(types)+"_"+str(j)
    latent_code_mean=dann.ae1.predict(expr)[0]
    np.savetxt("/data02/tguo/batch_effect/Pancreas/"+str(types1)+"_latent_dann.txt",latent_code_mean,fmt="%.5f")
    a=np.hstack((np.zeros((latent_code_mean.shape[0],2)),latent_code_mean[:,2:]))
    expr=np.hstack((data,a))
    correct=dann.ae1.predict(expr)[1]
    np.savetxt("/data02/tguo/batch_effect/Pancreas/"+str(types1)+"_correct_dann.txt",correct,fmt="%.5f")