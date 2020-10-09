import tensorflow as tf

class Dice(tf.keras.layers.Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        return self.alpha * (1.0 - x_p) * x + x_p * x

class dice(tf.keras.layers.Layer):
    def __init__(self, feat_dim):
        super(dice, self).__init__()
        self.feat_dim = feat_dim
        self.alphas= tf.Variable(tf.zeros([feat_dim]), dtype=tf.float32)
        self.beta  = tf.Variable(tf.zeros([feat_dim]), dtype=tf.float32)

        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, _x, axis=-1, epsilon=0.000000001):

        reduction_axes = list(range(len(_x.get_shape())))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(_x.get_shape())
        broadcast_shape[axis] = self.feat_dim

        mean = tf.reduce_mean(_x, axis=reduction_axes)
        brodcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
        std = tf.sqrt(std)
        brodcast_std = tf.reshape(std, broadcast_shape)

        x_normed = self.bn(_x)
        x_p = tf.keras.activations.sigmoid(self.beta * x_normed)

        return self.alphas * (1.0 - x_p) * _x + x_p * _x