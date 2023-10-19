import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# load minits dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# compute number of labels
num_labels = len(np.unique(y_train))

# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Image dimension
image_size =  x_train.shape[1]
input_size = image_size * image_size

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# network parameters
batch_size = 128
hidden_unit = 256
dropuout = 0.45

# 3-layer mlp with relu and dropout after each layer
model = Sequential()
model.add(Dense(hidden_unit, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropuout))
model.add(Dense(hidden_unit))
model.add(Activation('relu'))
model.add(Dropout(dropuout))
model.add(Dense(num_labels))
#this is the output for one-hot vector
model.add(Activation('softmax'))
model.summary()

plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

#validate model on validationset 
_, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))