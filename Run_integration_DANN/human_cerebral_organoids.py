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

DIR="/data02/tguo/batch_effect/harmony_data/cerebral_organoids_Kanton_2019_2m/"

data=np.loadtxt(DIR+"mat.txt",dtype=np.float64)
celltype=[]
with open(DIR+"celltype.txt") as inputfile:
    for line in inputfile:
        line=line.strip("\n")
        celltype.append(line)
celltype=np.array(celltype)
batch=np.loadtxt(DIR+"batch.txt",dtype=np.int32)
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

n = 5
for j in np.arange(n):
    K.clear_session()
    expr = data_train
    label = batch_train
    epoch = 500
    latent = 10
    class_dim = 2
    batch_size = 64
    lr = 0.001
    beta = 0
    gamma = 1
    alpha = 1
    fg_lambda = 0.1
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
    types = "latent_" + str(fg_lambda * 10) + "_" + str(j) + ".txt"
    np.savetxt(DIR+ str(types),
               latent_code_mean, fmt="%.5f")
    types = "correct_" + str(fg_lambda * 10) + "_" + str(j) + ".txt"
    np.savetxt(DIR+str(types), correct,
               fmt="%.5f")