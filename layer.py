import tensorflow as tf

class ArcMarginProduct_v2(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(ArcMarginProduct_v2, self).__init__()
        self.num_classes= num_classes
    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
    def call(self, input):
        cosine = tf.matmul(tf.nn.l2_normalize(input, axis=1), tf.nn.l2_normalize(self.w, axis=0))
        return cosine