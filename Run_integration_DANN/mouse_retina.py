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
from DANN.construct_network_2batch import VAE

DIR="/data02/tguo/batch_effect/mouse_retina/"
data=np.loadtxt(DIR+"data.txt",dtype=np.float64)
celltype=np.loadtxt(DIR+"celltype.txt",dtype=np.str)
batch=np.loadtxt(DIR+"batch.txt",dtype=np.int16)
data=minmax_scale(data,axis=0)
data=np.transpose(data)
ub=np.unique(batch)
num_1=np.where(batch==ub[0])[0]
num_2=np.where(batch==ub[1])[0]
batch1=np.zeros((len(batch),1))
batch1[num_1,0]=0
batch1[num_2,0]=1


####training set######
N=5000
random.shuffle(num_1)
random.shuffle(num_2)
num_train=np.hstack((num_1[range(N)],num_2[range(N)]))
data_train=data[num_train,:]
celltype_train=celltype[num_train]
batch1_train=batch1[num_train,:]

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
L=[0.1,0.5,1]
for j in np.arange(1,6):
    es=EarlyStopping(monitor='loss', patience=100, verbose=0, mode='auto')
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
    fg_lambda=0.1
    expr_label=np.hstack((expr,label,np.zeros((expr.shape[0],latent))))
    vae_=VAE(expr.shape[1],label.shape[1],latent,class_dim,lr,beta,gamma,alpha,fg_lambda)
    vae_.vae_build()
    myhistory=History()
    loss_=vae_.vae.fit(x=expr_label,epochs=epoch,batch_size=batch_size,shuffle=True,verbose=2,callbacks=[myhistory,es])
    total_loss=loss_.history['loss']
    for v in myhistory.history.values():
        plt.plot(v)
    expr_label_test=np.hstack((data,batch1,np.zeros((data.shape[0],latent))))
    latent_code_mean=vae_.ae1.predict(expr_label_test)[0]
    a=np.hstack((np.zeros((data.shape[0],class_dim)),latent_code_mean[:,class_dim:]))
    expr_label_1=np.hstack((data,batch1,a))
    correct=vae_.ae2.predict(expr_label_1)
    # np.savetxt("/data02/tguo/batch_effect/mouse_retina/data_"+str(j)+"_scidr.txt",latent_code_mean,fmt="%.5f")
    # np.savetxt("/data02/tguo/batch_effect/mouse_retina/correct_"+str(j)+"_scidr.txt",correct,fmt="%.5f")
#     np.savetxt("/data02/tguo/batch_effect/mouse_retina/loss_"+str(fg_lambda)+".txt",v,fmt='%.6f')