import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
import math
from keras.layers import Conv2D, MaxPooling2D, PReLU, Layer, AveragePooling2D
from keras import backend as K
from keras.models import Model
import tensorflow as tf

class ArcFaceLoss(Layer):
    def __init__(self,output_dim,scale = 128.,margin = 0.25,easy_margin=True,**kwargs):
        self.output_dim = output_dim
        # self.features = features
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        super(ArcFaceLoss, self).__init__(**kwargs)

    def build(self, input_shape):
    
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)

    def call(self,inputs):
        self.inputs = inputs
        print(inputs.shape)
        # self. embedding_dim = embedding_dim
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.threshold = math.cos(math.pi - self.margin)
        
        # self.var_weights = K.tf.Variable(keras.initializers.random_normal([inputs.shape[1]]), name='weights')
        # self.normed_weights = K.tf.nn.l2_normalize(self.var_weights, 1, 1e-10, name='weights_norm')
        self.normed_weights = K.tf.nn.l2_normalize(self.kernel, 0, 1e-10)
        # self.x_norm = K.tf.nn.l2_normalize(self.inputs, 1, 1e-10)
        self.normed_features = K.tf.nn.l2_normalize(self.inputs, 1, 1e-10, name='features_norm')

        self.cosine = tf.matmul(self.normed_features, self.normed_weights,transpose_a=False, transpose_b=False)
        self.logits = self.cosine
        return self.logits

    def compute_output_shape(self, input_shape):
        return (K.int_shape(self.logits))

    def loss(self,y_true,y_pred):
        # y_true = K.expand_dims(y_true[:, 0], 1)
        # y_true = K.cast(y_true, 'int32')
        # batch_idxs = K.arange(0, K.shape(y_true)[0])
        # batch_idxs = K.expand_dims(batch_idxs, 1)
        # ordinal_y = K.concatenate([batch_idxs, y_true], 1)
        # sel_logits = K.tf.gather_nd(self.logits, ordinal_y)

        # one_hot_mask = K.tf.one_hot(y_true, self.output_dim, on_value=1., off_value=0., axis=-1, dtype=tf.float32)

        cosine_theta_2 = K.tf.pow(y_pred, 2., name='cosine_theta_2')
        sine_theta = K.tf.pow(1. - cosine_theta_2, 0.5, name='sine_theta')

        if self.easy_margin:
            clip_mask = K.tf.to_float(y_pred >= 0.) * self.scale * y_pred * y_true#* one_hot_mask
        else:
            clip_mask = K.tf.to_float(y_pred >= self.threshold) * self.scale * self.mm*y_true #* one_hot_mask

        cosine_theta_m = self.scale * (self.cos_m * y_pred - self.sin_m *sine_theta)*y_true #* one_hot_mask

        updated_logits = self.scale * y_pred * (1.-y_true) + K.tf.where(clip_mask > 0., cosine_theta_m, clip_mask)
        # print("sddrs",one_hot_mask.shape)
        print("upda",updated_logits.shape)
        return K.categorical_crossentropy(y_true,updated_logits)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'margin':self.margin,
                  'easy_margin':self.easy_margin,
                  'scale': self.scale}
        base_config = super(ArcFaceLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))                                                                        