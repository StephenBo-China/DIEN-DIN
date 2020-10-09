import tensorflow as tf
from tensorflow.keras import layers

class AuxLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(layers.Dense(100, activation="sigmoid")) 
        self.fc.add(layers.ReLU())
        self.fc.add(layers.Dense(50, activation="sigmoid"))
        self.fc.add(layers.ReLU())
        self.fc.add(layers.Dense(2, activation=None))

    def call(self, input):
        logit = tf.squeeze(self.fc(input))
        return tf.keras.activations.softmax(logit)

