import tensorflow as tf
from keras.engine import Layer
import keras.backend as K

def zero_gradient(X):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        zero_gradient.num_calls += 1
    except AttributeError:
        zero_gradient.num_calls = 1

    grad_name = "GradientZero%d" % zero_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.identity(grad)*0.0]

    #g = K.get_session().graph
    g = tf.compat.v1.Session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientZero(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self,**kwargs):
        super(GradientZero, self).__init__(**kwargs)
        self.supports_masking = False

    def build(self, input_shape):
        self._trainable_weights = []

    def call(self, x, mask=None):
        return zero_gradient(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(GradientZero, self).get_config()
        return dict(list(base_config.items()))