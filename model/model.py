import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Encoders:
    class Encoder:
        def __init__(self, shape=None, layers=(16, 4)):
            self.shape = shape
            self.model = None
            self.layers = layers
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.float32)
            hidden = [input]
            for i in self.layers:
                hidden.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.softplus)(hidden[-1]))
            self.model = tf.keras.Model(inputs=[input], outputs=[hidden[-1]])

class NN:
    def __init__(self, encoders=[], layers=(16, 8, 4), default_activation=tf.keras.activations.softplus, mode='aleatoric'):
        self.encoders = encoders
        self.layers = layers
        self.mode = mode
        self.default_activation = default_activation
        self.model = None
        self.build()

    def build(self):
        inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.encoders]
        encodings = [encoder(input) for input, encoder in zip(inputs, self.encoders)]
        fused = [tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(encodings)]
        for i in self.layers:
            fused.append(tf.keras.layers.Dense(units=i, activation=self.default_activation)(fused[-1]))
        t = tf.keras.layers.Dense(units=2, activation=None)(fused[-1])
        if self.mode == 'aleatoric':
            self.model = tf.keras.Model(inputs=inputs, outputs=[t])
        else:
            self.model = tf.keras.Model(inputs=inputs, outputs=[t[:, 0]])

class Losses:
    class LogNormal(tf.keras.losses.Loss):
        def __init__(self, name):
            super(Losses.LogNormal, self).__init__(name=name)
        def call(self, y_true, y_pred):
            mu = y_pred[:, 0]
            std = tf.math.exp(y_pred[:, 1])
            cond_dist = tfp.distributions.LogNormal(loc=mu, scale=std)
            return -cond_dist.log_prob(y_true[:, 0])
        def __call__(self, y_true, y_pred, sample_weight=None):
            loss = self.call(y_true, y_pred)
            if sample_weight is not None:
                return tf.reduce_sum(loss * sample_weight, axis=0) / tf.reduce_sum(sample_weight, axis=0)
            else:
                return tf.reduce_mean(loss, axis=0)