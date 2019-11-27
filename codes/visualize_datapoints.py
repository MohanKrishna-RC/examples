import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
import math
from keras.layers import Conv2D, MaxPooling2D, PReLU, Layer, AveragePooling2D
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

class Histories(keras.callbacks.Callback):
    def __init__(self, loss, monitor):
        self.loss = loss
        self.path = os.path.join('images', self.loss)
        self.monitor = monitor
        if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf
        return

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            if self.monitor_op(current, self.best):
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f'
                      % (epoch + 1, self.monitor, self.best,
                         current))
                self.best = current
                self.model.save('best_model_with_' + self.loss + '.h5', overwrite=True)

        print('\n======================================')
        print('using loss type: {}'.format(self.loss))
        print(len(self.validation_data))  # be careful of the dimenstion of the self.validation_data, somehow some extra dim will be included
        print(self.validation_data[0].shape)
        print(self.validation_data[1].shape)
        print('========================================')
        # (IMPORTANT) Only use one input: "inputs=self.model.input[0]"
        nn_input = self.model.input  # this can be a list or a matrix.

        labels = self.validation_data[1].flatten()
        print(labels)
        feature_layer_model = Model(nn_input, outputs=self.model.get_layer('feature_embedding').output)
        feature_embedding = feature_layer_model.predict(self.validation_data[0])
        print(feature_embedding.shape)
        # if self.loss == 'AM-softmax':
        #     feature_embedding = tm.unit_vector(feature_embedding, axis=1)
        visualize(feature_embedding, labels, epoch, self.loss, self.path)
        return

    def on_batch_begin(self, batch, logs=None):
            return

    def on_batch_end(self, batch, logs=None):
        return

def visualize(feat, labels, epoch, loss, path):
    
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    XMax = np.max(feat[:, 0])
    XMin = np.min(feat[:, 1])
    YMax = np.max(feat[:, 0])
    YMin = np.min(feat[:, 1])

    plt.xlim(xmin=XMin, xmax=XMax)
    plt.ylim(ymin=YMin, ymax=YMax)
    plt.text(XMin, YMax, "epoch=%d" % epoch)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/' + loss + '_epoch=%d.png' % epoch)
