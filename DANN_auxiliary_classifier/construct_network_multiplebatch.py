import os
import tensorflow as tf
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config =  tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
tf.compat.v1.keras.backend.set_session(sess)

from keras.layers import Dense, Input, Dropout, Lambda, Layer, BatchNormalization, Activation
from keras import initializers, regularizers, metrics
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, EarlyStopping
from keras.utils import plot_model
import flip_gradient
import zero_gradient

tf.compat.v1.disable_eager_execution()


class DANN(object):
    def __init__(self, in_dim, in_category=1, latent_dim=10, class_dim=2, lr=0.001, beta=0, gamma=0, alpha=0,
                 fg_lambda=1, weight_class=1):
        self.latent = latent_dim
        self.in_dim = in_dim
        self.in_category = in_category
        self.class_dim = class_dim
        self.learning_rate = lr
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.fg_lambda = K.variable(K.cast_to_floatx(fg_lambda))
        self.weight_class = weight_class
        [self.weight, self.flip, self.vae, self.ae1] = self.vae_build()

    def vae_build(self):
        expr_in = Input(shape=(self.in_dim,))
        h0 = Dropout(0.5)(expr_in)
        in_dim = self.in_dim
        ##encoder
        h1 = Dense(units=256, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_initializer='zeros', kernel_regularizer=regularizers.l1(0.01), name='encoder1')(h0)
        h1 = BatchNormalization()(h1)
        h1 = Activation('relu')(h1)
        h2 = Dense(units=128, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_initializer='zeros', kernel_regularizer=None, name='encoder2')(h0)
        h2 = BatchNormalization()(h2)
        h2 = Activation('relu')(h2)
        h3 = Dense(units=64, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_initializer='zeros', kernel_regularizer=None, name='encoder3')(h0)
        h3 = BatchNormalization()(h3)
        h3 = Activation('relu')(h3)
        z = Dense(units=self.latent, activation=None,
                  kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                  bias_initializer='zeros', kernel_regularizer=None, name='latent_mean')(h3)
        #         z_mean=Dense(units=self.latent, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
        #                  bias_initializer='zeros', kernel_regularizer=None,name='latent_mean')(h3)
        #         z_mean=BatchNormalization()(z_mean)
        #         z_log_var=Dense(units=self.latent, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
        #                  bias_initializer='zeros', kernel_regularizer=None,name='latent_variation')(h3)
        #         z_log_var=BatchNormalization()(z_log_var)
        ##latent value
        #         z=Lambda(sampling, output_shape=(self.latent,))([z_mean,z_log_var])
        z_noise = Lambda(lambda x: x[:, :self.class_dim])(z)
        z_bio = Lambda(lambda x: x[:, self.class_dim:])(z)
        ###weight classifier#####
        wc_1 = zero_gradient.GradientZero()(z_bio)
        wc_2 = Dense(units=16, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                     bias_initializer='zeros', name='wc_1')(wc_1)
        wc_2 = BatchNormalization()(wc_2)
        wc_2 = Activation('relu')(wc_2)
        wc_3 = Dense(units=4, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                     bias_initializer='zeros', name='wc_2')(wc_2)
        wc_3 = BatchNormalization()(wc_3)
        #         wc_3=Activation('relu')(wc_3)
        wc_output = Dense(units=self.in_category, activation='softmax',
                          kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                          bias_initializer='zeros', name='weight_output')(wc_3)
        ###adversarial layer
        flip = flip_gradient.GradientReversal(self.fg_lambda)
        ad_h1 = flip(z_bio)
        ad_h2 = Dense(units=64, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                      bias_initializer='zeros', kernel_regularizer=None, name='ad_hidden1')(ad_h1)
        ad_h2 = BatchNormalization()(ad_h2)
        ad_h2 = Activation('relu')(ad_h2)
        ad_h3 = Dense(units=32, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                      bias_initializer='zeros', kernel_regularizer=None, name='ad_hidden2')(ad_h1)
        ad_h3 = BatchNormalization()(ad_h3)
        ad_h3 = Activation('relu')(ad_h3)
        ad_h4 = Dense(units=16, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                      bias_initializer='zeros', kernel_regularizer=None, name='ad_hidden3')(ad_h3)
        ad_h4 = BatchNormalization()(ad_h4)
        #         ad_h4=Activation('relu')(ad_h4)
        ad_output = Dense(units=self.in_category, activation='softmax',
                          kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                          bias_initializer='zeros', kernel_regularizer=None, name='bio_output')(ad_h4)
        ###noise classifier######
        nc_output = Dense(units=self.in_category, activation='softmax',
                          kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                          bias_initializer='zeros', name='noise_output')(z_noise)

        #####Decoder####
        decoder_h1 = Dense(units=64, activation=None,
                           kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                           bias_initializer='zeros', name='decoder_1')
        decoder_h2 = Dense(units=128, activation=None,
                           kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                           bias_initializer='zeros', name='decoder_2')
        decoder_h3 = Dense(units=256, activation='relu',
                           kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                           bias_initializer='zeros', name='decoder_3')
        recon_x = Dense(units=in_dim, activation='sigmoid', name='ae_output')
        dh1 = decoder_h1(z)
        dh1 = BatchNormalization()(dh1)
        dh1 = Activation('relu')(dh1)
        dh2 = decoder_h2(dh1)
        dh2 = BatchNormalization()(dh2)
        dh2 = Activation('relu')(dh2)
        dh3 = decoder_h3(dh2)
        dh3 = BatchNormalization()(dh3)
        dh3 = Activation('relu')(dh3)
        dh_output = recon_x(dh1)
        vae = Model(inputs=expr_in, outputs=[dh_output, nc_output, ad_output, wc_output])
        ae1 = Model(inputs=expr_in, outputs=[z, wc_output])
        ent = -K.sum(wc_output * K.log(wc_output), axis=1)
        return [ent / K.sum(ent, axis=0), flip, vae, ae1]

    def weighted_ad_class(self, y_true, y_pred):
        adversarial_loss = K.sum(K.categorical_crossentropy(y_true, y_pred) * self.weight)
        return adversarial_loss

    def compile(self, op):
        self.vae.compile(optimizer=op,
                         loss={'ae_output': 'binary_crossentropy', 'noise_output': 'categorical_crossentropy',
                               'bio_output': self.weighted_ad_class, 'weight_output': 'categorical_crossentropy'},
                         loss_weights={'ae_output': self.in_dim, 'noise_output': self.alpha, 'bio_output': self.gamma,
                                       'weight_output': self.weight_class})

    def train(self, trainX, trainY, epochs=1, batch_size=1, verbose=True, save_model=None):
        ad_loss = []
        wc_loss = []
        op = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #         op=SGD(lr=self.learning_rate)
        self.compile(op)
        for cnt in range(epochs):
            idx = np.arange(0, trainX.shape[0])
            random.shuffle(idx)
            trainX = trainX[idx, :]
            trainY = trainY[idx, :]
            N = trainX.shape[0] // batch_size
            p = np.float(cnt) / epochs
            # lr = 0.01 / (1. + 10 * p) ** 0.75
            # glr = (2. / (1. + np.exp(-10. * p)) - 1) * 0.1
            # K.set_value(self.vae.layers[7].hp_lambda, K.cast_to_floatx(glr))
            #             K.set_value(self.vae.optimizer.learning_rate, K.cast_to_floatx(lr))
            # Loop over each batch and train the model.
            #             self.compile(keras.optimizers.SGD(lr))
            for i in np.arange(N):
                batchX = trainX[(i * batch_size):((i + 1) * batch_size), :]
                batchY = trainY[(i * batch_size):((i + 1) * batch_size), :]
                metrics = self.vae.train_on_batch(x=batchX, y=[batchX, batchY, batchY, batchY])
            ad_loss.append(metrics[3])
            wc_loss.append(metrics[4])
            print(
                "Epoch {}/{}\n\t[total_loss: {:.5f}, vae_loss: {:.5f}, noise_loss: {:.5f}, bio_loss: {:.5f}, weight_loss: {:.5f}]".format(
                    cnt + 1, epochs, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
        return ad_loss + wc_loss
