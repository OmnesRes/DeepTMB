import tensorflow as tf
import tensorflow_probability as tfp

class Embed(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension=None, input_dimension=None, trainable=False, regularization=0):
        super(Embed, self).__init__()
        self.input_dimension = input_dimension
        self.embedding_dimension = embedding_dimension
        self.trainable = trainable
        self.regularization = regularization

    def build(self, input_shape):
        if self.input_dimension:
            self.embedding_matrix = self.add_weight(shape=[self.input_dimension, self.embedding_dimension], initializer="uniform", trainable=self.trainable, dtype=tf.float32, regularizer=tf.keras.regularizers.l2(self.regularization))
        else:
            self.embedding_matrix = self.add_weight(shape=[self.embedding_dimension, self.embedding_dimension], initializer=tf.keras.initializers.identity(), trainable=self.trainable, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.gather(tf.concat([tf.zeros([1, self.embedding_dimension]), self.embedding_matrix], axis=0), inputs, axis=0)

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

    class Embedder:
        def __init__(self, shape=None, dim=None, input_dim=None, layers=(16, 4)):
            self.shape = shape
            self.input_dim = input_dim
            self.dim = dim
            self.layers = layers
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(self.shape, dtype=tf.int32)
            hidden = [Embed(input_dimension=self.input_dim, embedding_dimension=self.dim)(input)]
            for i in self.layers:
                hidden.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.softplus)(hidden[-1]))

            self.model = tf.keras.Model(inputs=[input], outputs=[hidden[-1]])



class NN:
    def __init__(self, encoders=[], layers=(16, 8, 4), default_activation=tf.keras.activations.softplus, mode='johnson'):
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
        for index, i in enumerate(self.layers):
            fused.append(tf.keras.layers.Dense(units=i, activation=self.default_activation)(fused[-1]))
        if self.mode == 'johnson':
            fused.append([tf.keras.layers.Dense(units=1, activation=None)(fused[-1]),
                          tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softplus)(fused[-1]),
                          tf.keras.layers.Dense(units=1, activation=None)(fused[-1]),
                          tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softplus)(fused[-1])])
            t = tfp.layers.DistributionLambda(lambda t: tfp.distributions.JohnsonSU(skewness=t[0][:, 0], tailweight=t[1][:, 0], loc=t[2][:, 0], scale=t[3][:, 0]))(fused[-1])
        elif self.mode == 'mixture':
            fused.append([tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softmax)(fused[-1]),
                  tf.keras.layers.Dense(units=3, activation=None)(fused[-1]),
                  tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softplus)(fused[-1])])
            t = tfp.layers.DistributionLambda(lambda ts: tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(probs=ts[0]),
                                                                                              components_distribution=tfp.distributions.LogNormal(loc=ts[1], scale=ts[2])))(fused[-1])
        elif self.mode == 'normal':
            fused.append([tf.keras.layers.Dense(units=1, activation=None)(fused[-1]),
                  tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softplus)(fused[-1])])
            t = tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[0][:, 0], scale=t[1][:, 0]))(fused[-1])

        elif self.mode == 'tfp_linear_regresion':
            fused.append([tf.keras.layers.Dense(units=1, activation=None)(fused[-1]),
                  tfp.layers.VariableLayer(shape=(1, ), activation=tf.keras.activations.exponential)(fused[-1])])
            t = tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[0][:, 0], scale=t[1]))(fused[-1])

        else:
            t = tf.keras.layers.Dense(units=1, activation=None)(fused[-1])
        self.model = tf.keras.Model(inputs=inputs, outputs=[t])
