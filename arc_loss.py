import keras
import keras.backend as K
from keras.layers import Layer

import math

class ArcFaceLoss(Layer):

    def __init__(self, output_dim, initializer='uniform', scale=64., margin=0.5,
                 easy_margin=True,**kwargs):
        if margin < 0 or margin >= (math.pi/2.0):
            raise ValueError("allowed range for margin is : [0,pi/2)")
        self.output_dim = output_dim
        self.scale = scale
        self.margin = margin
        self.initializer = initializer
        self.easy_margin = easy_margin
        self.result = None
        super(ArcFaceLoss, self).__init__(**kwargs)

    def build(self, input_shape):        
        # Create a trainable weight variable for this layer.
        print(input_shape[0])
        print(input_shape[1])
        self.kernel = self.add_weight(name='weights', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.initializer,
                                      trainable=True)

    def call(self, input):
        """inputs[0] : features
           inputs[1] : embeddings
        """       
        normed_weights = K.tf.nn.l2_normalize(self.kernel,
                                              axis=1,
                                              epsilon=1e-10,
                                              name='weights_norm')
        normed_features = K.tf.nn.l2_normalize(input,
                                             axis=1,
                                             epsilon=1e-10,
                                             name='features_norm')
        cosine = K.tf.matmul(normed_features, normed_weights, 
                           transpose_a=False, transpose_b=False)        
        self.result = cosine
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
    def loss(self, y_true,y_pred):
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        mm = math.sin(math.pi - self.margin) * self.margin
        threshold = math.cos(math.pi - self.margin)
        
        cosine_theta_2 = K.tf.pow(y_pred, 2., name='cosine_theta_2')
        print(cosine_theta_2)
        sine_theta = K.tf.pow(1. - cosine_theta_2, .5, name='sine_theta')
        
        cosine_theta_m = self.scale * (cos_m * y_pred - sin_m * sine_theta) * y_true
        if self.easy_margin:
            clip_mask = K.tf.to_float(y_pred >= 0.) * self.scale * y_pred * y_true
        else:
            clip_mask = K.tf.to_float(y_true >= threshold) * self.scale * mm * y_true

        cosine = self.scale * y_pred * (1. - y_true) + K.tf.where(clip_mask > 0., cosine_theta_m, clip_mask)
        return K.categorical_crossentropy(y_true,cosine)