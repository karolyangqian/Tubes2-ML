import tensorflow as tf

class Conv2DScratch(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(Conv2DScratch, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True
        )
        super(Conv2DScratch, self).build(input_shape)