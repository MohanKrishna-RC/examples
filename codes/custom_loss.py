import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
import math
from keras.layers import Conv2D, MaxPooling2D, PReLU, Layer, AveragePooling2D
from keras import backend as K
import tensorflow as tf

class Dense_with_AMsoftmax_loss(Layer):
    
    def __init__(self, output_dim, m, scale, **kwargs):
        self.output_dim = output_dim
        self.m = m
        self.scale = scale
        super(Dense_with_AMsoftmax_loss, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape[1])
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)

    def call(self, inputs):
        self.inputs = inputs
        # self.xw = K.dot(inputs, self.kernel)
        # self.w_norm = K.tf.norm(self.kernel, axis=0) + 1e-8
        # self.x_norm = K.tf.norm(inputs, axis=1) + 1e-8
        # self.x_norm = K.expand_dims(self.x_norm, 1)
        # self.logits = self.xw / self.w_norm / self.x_norm
        self.w_norm = K.tf.nn.l2_normalize(self.kernel, 0, 1e-10)
        self.x_norm = K.tf.nn.l2_normalize(self.inputs, 1, 1e-10)
        self.logits = K.tf.matmul(self.x_norm, self.w_norm)

        return self.logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, y_true, y_pred):
        y_true = K.expand_dims(y_true[:, 0], 1)
        y_true = K.cast(y_true, 'int32')
        batch_idxs = K.arange(0, K.shape(y_true)[0])
        batch_idxs = K.expand_dims(batch_idxs, 1)
        ordinal_y = K.concatenate([batch_idxs, y_true], 1)
        sel_logits = K.tf.gather_nd(self.logits, ordinal_y)
        comb_logits_diff = K.tf.add(self.logits, K.tf.scatter_nd(ordinal_y, sel_logits - self.m - sel_logits, K.tf.to_int32(K.tf.shape(self.logits))))
        return K.sparse_categorical_crossentropy(y_true, self.scale * comb_logits_diff, from_logits=True)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'm': self.m,
                  'scale': self.scale}
        base_config = super(Dense_with_AMsoftmax_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
