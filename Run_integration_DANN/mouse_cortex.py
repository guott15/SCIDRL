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

DIR="/data02/tguo/batch_effect/mouse_cortex/"
types="allsame"
data=np.loadtxt(DIR+str(types)+"_mat.txt",dtype=np.float64)
celltype=pd.read_csv(DIR+str(types)+"_celltype.txt",sep=",",header=None).values[:,0]
batch=np.loadtxt(DIR+str(types)+"_batch.txt",dtype=np.int16)
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

n = 5
for j in np.arange(n):
    K.clear_session()
    expr = train_data
    label = train_batch
    epoch = 1000
    latent = 10
    class_dim = 2
    batch_size = 64
    lr = 0.001
    beta = 0
    gamma = 1
    alpha = 1
    fg_lambda = 10
    expr_label = np.hstack((expr, label, np.zeros((expr.shape[0], latent))))
    vae_ = VAE(expr.shape[1], label.shape[1], latent, class_dim, lr, beta, gamma, alpha, fg_lambda)
    vae_.vae_build()
    myhistory = History()
    loss_ = vae_.vae.fit(x=expr_label, epochs=epoch, batch_size=batch_size, shuffle=True, verbose=2,
                         callbacks=[myhistory])
    total_loss = loss_.history['loss']

    ###latent code
    expr_label_test = np.hstack((data, batch, np.zeros((data.shape[0], latent))))
    latent_code_mean = vae_.ae1.predict(expr_label_test)[0]
    a = np.hstack((np.zeros((data.shape[0], class_dim)), latent_code_mean[:, class_dim:]))
    expr_label_1 = np.hstack((data, batch, a))
    correct = vae_.ae1.predict(expr_label_1)[3]
    types = "allsame_" + str(fg_lambda * 10) + "_" + str(j)
    np.savetxt(DIR + str(types) + "_latent_scidr.txt", latent_code_mean,
               fmt="%.5f")
    np.savetxt(DIR + str(types) + "_correct_scidr.txt", correct, fmt="%.5f")

