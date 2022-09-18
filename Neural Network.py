#%% Libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras 
import matplotlib.pyplot as plt

#%% Load a predefined dataset 70k images of 28*28, remaining 10k for test
fashion_mnist = keras.datasets.fashion_mnist

# Pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Show data
# print(train_labels[0])
# print(train_images[0])

plt.imshow(train_images[2000], cmap = 'gray', vmin = 0, vmax = 255 )
plt.show()

#%% Define our neural net structure: less layers is better bcs it is easier to work
model = keras.Sequential([
    # input layer: flatten => does the multiplication and convert it into a single 784x1 pixel
    # it helps symplifing the data in the neural net
    keras.layers.Flatten(input_shape = (28,28)), # input layer, flatter => does the multiplication and convert it into a single 784x1 pixel
    # it helps symplifing the data in the neural net

    # hidden layer: is 128 deep. relu returns the value, or 0 (works good enough, much faster)
    keras.layers.Dense(units = 128, activation = tf.nn.relu),

    # output layer: is 0-10 (depending on what piece of clothing it is), return maximum
    # 10 because the keras data has 10 elements
    keras.layers.Dense(units = 10, activation = tf.nn.softmax) # softmax: takes the greatest number 
])

# Compile our model: loss => optimizer
model.compile(optimizer = tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics =('accuracy')) # aparse_... : tell you how right/wrong you are from 0 to 1

#%% Train our model, using our training data
model.fit(train_images, train_labels, epochs = 5)

# Test our model, using our testing data
test_loss = model.evaluate(test_images, test_labels)

plt.imshow(test_images[0], cmap = 'gray', vmin = 0, vmax = 255)
plt.show()

# Print label
print(test_labels[0]) # The number indicates the probability for the image number

# Make predictions
predictions = model.predict(test_images)
print(predictions[0])

# Print out predictions
print(list(predictions[1]).index(max(predictions[1])))

# Print the correct answer
# print(test_labels[1])


