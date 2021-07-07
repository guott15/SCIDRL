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
from copy import deepcopy
import tensorflow as tf
import keras
from keras.callbacks import Callback,EarlyStopping

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

k=1
data=np.loadtxt("/data02/tguo/batch_effect/simulate/dataset4-2_1same_group"+str(k)+"_data.txt",dtype=np.float64)
celltype=np.loadtxt("/data02/tguo/batch_effect/simulate/dataset4-2_1same_group"+str(k)+"_celltype.txt",dtype=np.str)
batch=np.loadtxt("/data02/tguo/batch_effect/simulate/dataset4-2_1same_group"+str(k)+"_batch.txt",dtype=np.int16)
data=minmax_scale(data, axis=0)
data=np.transpose(data)
N=np.min(batch)
if N>0:
    batch=batch-N

n=5
###清除现有的模型
for j in np.arange(n):
    ###清除现有的模型
    K.clear_session()
    expr=data
    label=batch.reshape(len(batch),1)
    epoch=5000
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
    loss_=vae_.vae.fit(x=expr_label,epochs=epoch,batch_size=batch_size,shuffle=True,verbose=2,callbacks=[myhistory])
    total_loss=loss_.history['loss']
    ###loss
    for v in myhistory.history.values():
        plt.plot(v)
    plt.plot(total_loss)
    plt.legend(list(myhistory.history.keys())+['tota_loss'], loc='upper right')
    plt.show()
    ###latent code
    latent_code_mean=vae_.ae1.predict(expr_label)[0]
    recon=vae_.ae1.predict(expr_label)[1]
    a=np.hstack((np.zeros((data.shape[0],class_dim)),latent_code_mean[:,class_dim:]))
    expr_label_1=np.hstack((data,label,a))
    correct=vae_.ae2.predict(expr_label_1)
    ad_h3=vae_.ae1.predict(expr_label)[2]
    np.savetxt("/data02/tguo/batch_effect/simulate/dataset4-2_1same_group" + str(k) + "_data_scidr_" + str(j) + ".txt",
               latent_code_mean, fmt="%.5f")
