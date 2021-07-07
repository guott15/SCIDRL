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
import tensorflow as tf
import keras
from keras.callbacks import Callback,EarlyStopping
from DANN.construct_network_multiplebatch import VAE

class History(Callback):
    def on_train_begin(self, logs={}):
        self.history = {}
        self.epoch = []
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        # Get the output layer names (we only have loss layers as output)
        output_layer_names = [output_layer.name for output_layer in self.model.output]
        # Get the mean error over all batches in this epoch
        output_layer_values = np.mean(self.model.predict(expr_label), axis = 1)
        # Store it to the history
        for k, v in zip(output_layer_names, output_layer_values):
            self.history.setdefault(k, []).append(v)
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return
es=EarlyStopping(monitor='loss', patience=100, verbose=0, mode='auto')

DIR="/data02/tguo/batch_effect/"

data1=np.loadtxt(DIR+"kidney_mat.txt",dtype=np.float64)
data2=np.loadtxt(DIR+"liver_mat.txt",dtype=np.float64)
data3=np.loadtxt(DIR+"eso_mat.txt",dtype=np.float64)
data4=np.loadtxt(DIR+"lung_mat.txt",dtype=np.float64)
data5=np.loadtxt(DIR+"spleen_mat.txt",dtype=np.float64)
data6=np.loadtxt(DIR+"pbmc1_mat.txt",dtype=np.float64)
data7=np.loadtxt(DIR+"pbmc2_mat.txt",dtype=np.float64)
data8=np.loadtxt(DIR+"pan_mat.txt",dtype=np.float64)

celltype1=pd.read_csv(DIR+"kidney_celltype.txt",sep=",",header=None).values[:,0]
celltype2=pd.read_csv(DIR+"liver_celltype.txt",sep=",",header=None).values[:,0]
celltype3=pd.read_csv(DIR+"eso_celltype.txt",sep=",",header=None).values[:,0]
celltype4=pd.read_csv(DIR+"lung_celltype.txt",sep=",",header=None).values[:,0]
celltype5=pd.read_csv(DIR+"spleen_celltype.txt",sep=",",header=None).values[:,0]
celltype6=pd.read_csv(DIR+"pbmc1_celltype.txt",sep=",",header=None).values[:,0]
celltype7=pd.read_csv(DIR+"pbmc2_celltype.txt",sep=",",header=None).values[:,0]
celltype8=pd.read_csv(DIR+"pan_celltype.txt",sep=",",header=None).values[:,0]

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
k=3
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

n=5
for j in np.arange(n):
    K.clear_session()
    expr=data_train
    label=batch_train
    epoch=500
    latent=10
    class_dim=2
    batch_size=64
    lr=0.001
    beta=0
    gamma=1
    alpha=1
    fg_lambda=0.1
    expr_label=np.hstack((expr,label,np.zeros((expr.shape[0],latent))))

    vae_=VAE(expr.shape[1],label.shape[1],latent,class_dim,lr,beta,gamma,alpha,fg_lambda)
    vae_.vae_build()
    myhistory=History()
    loss_=vae_.vae.fit(x=expr_label,epochs=epoch,batch_size=batch_size,shuffle=True,verbose=2,callbacks=[myhistory,es])
    total_loss=loss_.history['loss']
    ###loss
    for v in myhistory.history.values():
        plt.plot(v)
    plt.plot(total_loss)
    plt.legend(list(myhistory.history.keys())+['tota_loss'], loc='upper right')
    plt.show()
    ###latent code
    expr_label_test=np.hstack((data,batch,np.zeros((data.shape[0],latent))))
    latent_code_mean=vae_.ae1.predict(expr_label_test)[0]
    # recon=vae_.ae1.predict(expr_label)[1]
    # a=np.hstack((np.zeros((data.shape[0],class_dim)),latent_code_mean[:,class_dim:]))
    # expr_label_1=np.hstack((data,batch1,a))
    # correct=vae_.ae2.predict(expr_label_1)
    # ad_h3=vae_.ae1.predict(expr_label)[2]
    # N=len(v)
    # convergent_value[k,j-1]=np.mean(v[N-20:N])
    umap_mat=umap.UMAP().fit_transform(latent_code_mean[:,class_dim:])
    np.savetxt(DIR+"allorgan/mat_"+str(fg_lambda*10)+"_"+str(j)+"_scidr.txt",latent_code_mean,fmt="%.5f")
    np.savetxt(DIR+"celltype_"+str(fg_lambda*10)+"_"+str(j)+"_scidr.txt",celltype,fmt="%s")
    np.savetxt(DIR+"batch_"+str(fg_lambda*10)+"_"+str(j)+"_scidr.txt",batch,fmt="%d")
    np.savetxt(DIR+"mat_umap_"+str(fg_lambda*10)+"_"+str(j)+"_scidr.txt",umap_mat,fmt="%.5f")
    np.savetxt(DIR+"loss_"+str(fg_lambda*10)+"_"+str(j)+"_scidr.txt",np.array(v),fmt="%.5f")