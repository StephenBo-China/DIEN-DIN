import tensorflow as tf
from tensorflow.keras import layers
from activations import Dice,dice

class GRU_GATES(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GRU_GATES, self).__init__()
        self.linear_act = layers.Dense(units, activation=None, use_bias=True)
        self.linear_noact = layers.Dense(units, activation=None, use_bias=False)

    def call(self, a, b, gate_b=None):
        if gate_b is None:
            return tf.keras.activations.sigmoid(self.linear_act(a) + self.linear_noact(b))
        else:
            return tf.keras.activations.tanh(self.linear_act(a) + tf.math.multiply(gate_b, self.linear_noact(b)))

class AUGRU(layers.Layer):
    def __init__(self, units):
        super(AUGRU, self).__init__()
        self.u_gate = GRU_GATES(units)
        self.r_gate = GRU_GATES(units)
        self.c_memo = GRU_GATES(units)

    def call(self, inputs, state, att_score):
        u = self.u_gate(inputs, state) #u_t
        r = self.r_gate(inputs, state) #r_t
        c = self.c_memo(inputs, state, r) #\tilde{h_t}
        u_= att_score * u #\tilde{u_{t}'} [AUGRU Add]
        state_next = (1 - u_) * state + u_ * c #h_t [AUGRU change u_t on output]
        return state_next

class attention(tf.keras.layers.Layer):
    def __init__(self, keys_dim):
        super(attention, self).__init__()
        self.keys_dim = keys_dim
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(layers.Dense(36, activation="sigmoid")) 
        self.fc.add(dice(36))
        self.fc.add(layers.Dense(1, activation=None))

    def call(self, queries, keys, keys_length):
        #Attention
        queries = tf.tile(tf.expand_dims(queries, 1), [1, tf.shape(keys)[1], 1])
        din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
        outputs = tf.transpose(self.fc(din_all), [0,2,1])
        key_masks = tf.sequence_mask(keys_length, max(keys_length), dtype=tf.bool)
        key_masks = tf.expand_dims(key_masks, 1)
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)
        outputs = outputs / (self.keys_dim ** 0.5)
        #outputs = tf.keras.activations.softmax(outputs, -1)
        outputs = tf.keras.activations.sigmoid(outputs)
        
        #Sum Pooling
        outputs = tf.squeeze(tf.matmul(outputs, keys))
        print("outputs:" + str(outputs.numpy().shape))
        return outputs