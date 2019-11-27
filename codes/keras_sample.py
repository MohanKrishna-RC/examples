from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from arcloss_custom import ArcFaceLoss
from visualize_datapoints import Histories
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda
from keras.models import Model
from custom_loss import Dense_with_AMsoftmax_loss
# import keras.backend.tensorflow_backend as K
# K.set_session

batch_size = 4
num_classes = 10
epochs = 50

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_val, y_val) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')/255.0
x_val = x_val.astype('float32')/255.0
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_val.shape[0], 'test samples')

x_train = x_train[0:50]
y_train = y_train[0:50]
print(x_train.shape)
print(y_train.shape)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# model = Sequential()

input_shape = (28,28,1)
nn_inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(nn_inputs)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

feature_embedding = Dense(64, name='feature_embedding')(x)
# model.add(Conv2D(32, Kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# # model.add(Conv2D(64, (3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# out = Dense(num_classes, activation='softmax')(feature_embedding)
# model = Model(inputs=nn_inputs, outputs=out)

center_logits = ArcFaceLoss(num_classes,margin=0.2, scale=24)
print(center_logits)
out = center_logits(feature_embedding)
model = Model(inputs=nn_inputs, outputs=out)
model.summary()
loss_name = 'ArcFaceLoss'
optimizer = keras.optimizers.Adam(lr = 0.01,clipnorm = 1.)
model.compile(loss=center_logits.loss,
              optimizer='Adam',
              metrics=['accuracy'])
histories = Histories(loss=loss_name, monitor='val_loss')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val[0:30], y_val[0:30]))
          #callbacks=[histories])
# score = model.evaluate(x_val, y_val, verbose=0)

model.save("model.h5")
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])