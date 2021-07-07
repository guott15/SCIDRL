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

DIR="/data02/tguo/batch_effect/cell_lines/"
data=np.loadtxt(DIR+"data.txt",dtype=np.float64)
celltype=np.loadtxt(DIR+"celltype.txt",dtype=np.str)
batch=np.loadtxt(DIR+"batch.txt",dtype=np.str)
data=minmax_scale(data,axis=1)
nb=np.unique(batch)
batch1=np.zeros((len(batch),len(nb)))
for i in np.arange(len(nb)):
    batch1[np.where(batch==nb[i])[0],i]=1

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

###清除现有的模型
L=np.arange(0.1,1.1,0.1)
Loss=np.zeros(len(L))
j=0
for l in [0.1]:
    K.clear_session()
    expr=data
    label=batch1
    epoch=500
    latent=10
    class_dim=2
    batch_size=64
    lr=0.001
    beta=0
    gamma=1
    alpha=1
    fg_lambda=l
    expr_label=np.hstack((expr,label,np.zeros((expr.shape[0],latent))))

    vae_=VAE(expr.shape[1],label.shape[1],latent,class_dim,lr,beta,gamma,alpha,fg_lambda)
    vae_.vae_build()
    myhistory=History()
    loss_=vae_.vae.fit(x=expr_label,epochs=epoch,batch_size=batch_size,shuffle=True,verbose=2,callbacks=[myhistory])
    total_loss=loss_.history['loss']
    ###loss 
    for v in myhistory.history.values():
        plt.plot(v)
    plt.plot(total_loss)
    plt.legend(list(myhistory.history.keys())+['tota_loss'], loc='upper right')
    plt.show()
#     Loss[j]=np.mean(v[(len(v)-20):len(v)])
#     j=j+1
#     ###latent code
    latent_code_mean=vae_.ae1.predict(expr_label)[0]