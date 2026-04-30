import tensorflow as tf

class SimpleRNNScratch(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleRNNScratch, self).__init__(**kwargs)