import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical

#load dataset from mnist library
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Total number of labels in the train_data
num_labels = len(np.unique(y_train))

# One-hot encoding of labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#reshape and normalize the input
image_shape = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_shape, image_shape, 1])
x_train = x_train.astype('float32') / 255

x_test = np.reshape(x_test, [-1, image_shape, image_shape, 1])
x_test = x_test.astype('float32') / 255

# Define network parameters
input_shape = (image_shape, image_shape, 1)
num_filters = 64
batch_size = 128
dropout = 0.3
kernel_size = 3

# Using functional API to develop model
inputs = Input(shape=input_shape)
x = Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu')(inputs)
x = MaxPooling2D()(x)
x = Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu')(x)
x = Flatten()(x)
x = Dropout(dropout)(x)
outputs = Dense(num_labels, activation='softmax')(x)

# build model
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=batch_size)

score = model.evaluate(x_test, y_test, batch_size, verbose=0)