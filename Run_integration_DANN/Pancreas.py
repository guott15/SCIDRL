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

DIR="/data02/tguo/batch_effect/Pancreas/baron_muraro_"
types='8same-1'
data=np.loadtxt(DIR+str(types)+"_data.txt", dtype=np.float64)
celltype=np.loadtxt(DIR+str(types)+"_celltype.txt", dtype=np.str)
batch=np.loadtxt(DIR+str(types)+"_batch.txt", dtype=np.int16)
data=minmax_scale(data,feature_range=(0,1),axis=1)

n=5
for j in np.arange(n):
    K.clear_session()
    expr=data
    label=batch
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
    latent_code_mean=vae_.ae1.predict(expr_label)[0]
    a=np.hstack((np.zeros((expr.shape[0],class_dim)),latent_code_mean[:,class_dim:]))
    expr_label_1=np.hstack((expr,label,a))
    correct=vae_.ae2.predict(expr_label_1)
    types1=str(types)+'_'+str(fg_lambda*10)+"_"+str(j)
    np.savetxt(DIR+str(types1)+"_SCIDR_latent.txt",latent_code_mean,fmt="%.5f")
    np.savetxt(DIR+str(types1)+"_SCIDR_correct.txt",correct,fmt="%.5f")
