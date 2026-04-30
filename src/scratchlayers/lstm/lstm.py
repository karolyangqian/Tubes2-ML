import tensorflow as tf

class LSTMScratch(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LSTMScratch, self).__init__(**kwargs)