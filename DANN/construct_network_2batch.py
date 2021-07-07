##################autoencoder#####################
import flip_gradient
import tensorflow as tf
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
from keras.layers import Dense, Input, Dropout, Lambda, Layer, BatchNormalization, Activation
from keras import initializers, regularizers, metrics
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras.utils import plot_model
tf.compat.v1.disable_eager_execution()

def sampling(args):
    epsilon_std = 1.0
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=epsilon_std)
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z


class VAE():
    def __init__(self, in_dim, in_category=1, latent_dim=10, class_dim=2, lr=0.001, beta=0, gamma=0, alpha=0,
                 fg_lambda=1):
        self.latent = latent_dim
        self.in_dim = in_dim
        self.in_category = in_category
        self.class_dim = class_dim
        self.learning_rate = lr
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.fg_lambda = fg_lambda
        self.vae = None
        self.ae = None

    def vae_build(self):
        INPUTS = Input(shape=(self.in_dim + self.in_category + self.latent,))
        expr_in = Lambda(lambda x: x[:, :self.in_dim])(INPUTS)
        batch_in = Lambda(lambda x: x[:, self.in_dim:(self.in_dim + self.in_category)])(INPUTS)
        latent_in = Lambda(lambda x: x[:, (self.in_dim + self.in_category):])(INPUTS)
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
        ad_h4 = Activation('relu')(ad_h4)
        ad_output = Dense(units=self.in_category, activation='softmax',
                          kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                          bias_initializer='zeros', kernel_regularizer=None, name='ad_output')(ad_h4)
        ###classifier layer
        class_output = Dense(units=self.in_category, activation='softmax',
                             kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                             bias_initializer='zeros', name='classifier_final')(z_noise)
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
        recon_x = Dense(units=in_dim, activation='sigmoid')

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
        ####correct output###
        cdh1 = decoder_h1(latent_in)
        cdh1 = BatchNormalization()(cdh1)
        cdh1 = Activation('relu')(cdh1)
        cdh2 = decoder_h2(cdh1)
        cdh2 = BatchNormalization()(cdh2)
        cdh2 = Activation('relu')(cdh2)
        cdh3 = decoder_h3(cdh2)
        cdh3 = BatchNormalization()(cdh3)
        cdh3 = Activation('relu')(cdh3)
        cdh_output = recon_x(cdh1)

        ##计算loss层
        class CoustomLossLayerReconstruction(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                self.in_dim = in_dim
                super(CoustomLossLayerReconstruction, self).__init__(**kwargs)

            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean = inputs[1]
                xent_loss = in_dim * metrics.binary_crossentropy(x, x_decoded_mean)
                #                 xent_loss=K.sum((x-x_decoded_mean)**2,axis=-1)/in_dim
                self.add_loss(K.mean(xent_loss), inputs=inputs)
                return xent_loss

        class CoustomLossLayerKL(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CoustomLossLayerKL, self).__init__(**kwargs)

            def call(self, inputs):
                z_mean = inputs[0]
                z_log_var = inputs[1]
                kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                self.add_loss(beta * K.mean(kl_loss), inputs=inputs)
                return kl_loss

        class CoustomLossLayerAdversarial(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CoustomLossLayerAdversarial, self).__init__(**kwargs)

            def call(self, inputs):
                y = inputs[0]
                y_pred = inputs[1]
                #                 adversarial_loss=metrics.binary_crossentropy(y, y_pred)
                adversarial_loss = metrics.categorical_crossentropy(y, y_pred)
                self.add_loss(alpha * K.mean(adversarial_loss), inputs=inputs)
                return adversarial_loss

        class CoustomLossLayerClassifier(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CoustomLossLayerClassifier, self).__init__(**kwargs)

            def call(self, inputs):
                y = inputs[0]
                y_pred = inputs[1]
                #                 classifier_loss=metrics.binary_crossentropy(y, y_pred)
                classifier_loss = metrics.categorical_crossentropy(y, y_pred)
                self.add_loss(gamma * K.mean(classifier_loss), inputs=inputs)
                return classifier_loss

        y_ent = CoustomLossLayerReconstruction(name="myCustomLossLayerReconstruction")([expr_in, dh_output])
        #         y_kl=CoustomLossLayerKL(name="myCustomLossLayerKL")([z_mean, z_log_var])
        y_classifier = CoustomLossLayerClassifier(name="myCoustomLossLayerClassifier")([batch_in, class_output])
        y_adversarial = CoustomLossLayerAdversarial(name="myCoustomLossLayerAdversarial")([batch_in, ad_output])
        #         vae=Model(inputs=INPUTS, outputs=[y_kl,y_ent,y_classifier,y_adversarial])
        #         ae1=Model(inputs=INPUTS,outputs=[z_mean,dh_output,ad_h2])
        vae = Model(inputs=INPUTS, outputs=[y_ent, y_classifier, y_adversarial])
        ae1 = Model(inputs=INPUTS, outputs=[z, dh_output, ad_h2, cdh_output])
        op = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        vae.compile(optimizer=op, loss=None)
        self.vae = vae
        self.ae1 = ae1