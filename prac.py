import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import glorot_normal, normal

# ======================= #
# Data generation process #
# ======================= #

# feature x1, 1,000 points evenly spaced between -1 and 1
x1 = np.linspace(-1, 1, 1000)
plt.plot(x1)
plt.show()
# feature x2, for the two curves
x2_blue = np.square(x1)
x2_green = np.square(x1) - 0.5

# coordinates for points in the blue line
blue_line = np.vstack([x1, x2_blue])
# coordinates for points in the green line
green_line = np.vstack([x1, x2_green])

# Remember, blue line is negative (0) and green line is positive (1)
X = np.concatenate([blue_line, green_line], axis=1)
y = np.concatenate([np.zeros(1000), np.ones(1000)])

# But we must not feed the network with neatly organized inputs...
# so let's randomize them using our lucky number 13!
np.random.seed(13)
shuffled = np.random.permutation(range(X.shape[1]))
X = X.transpose()[shuffled]
y = y[shuffled].reshape(-1, 1)

# ==================== #
# Building the network #
# ==================== #

# Choose an activation function for the hidden layer: sigmoid, tanh or relu
activation = 'sigmoid'

# Uses Glorot initializer for hidden layer with a typical seed: 42
glorot_initializer = glorot_normal(seed=42)
# Uses Normal initializer for outputlayer with the same seed
normal_initializer = normal(seed=42)

# Uses Stochastic Gradient Descent with a learning rate of 0.05
sgd = SGD(lr=0.05)

# Uses Keras' Sequential API
model = Sequential()

model.add(Dense(input_dim=2, # Input layer contains 2 units
                units=2,     # Hidden layer contains 2 units
                kernel_initializer=glorot_initializer, 
                activation=activation))

# Output layer with sigmoid activation for binary classification
model.add(Dense(units=1, 
                kernel_initializer=normal_initializer,
                activation='sigmoid'))

# Compiles model using binary crossentropy as loss
model.compile(loss='binary_crossentropy', 
              optimizer=sgd, 
              metrics=['acc'])

# Fits the model using a mini-batch size of 16 during 150 epochs
model.fit(X, y, epochs=150, batch_size=16)

