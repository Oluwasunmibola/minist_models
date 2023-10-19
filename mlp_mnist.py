import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

def dataset_checkout():
    #load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Count number of unique train labels
    unique, counts = np.unique(y_train, return_counts=True)
    print("Train labels: ", dict(zip(unique, counts)))

    #Count number of unique test labels
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    print("Test labels: ", dict(zip(test_unique, test_counts)))

    #Sample 25 mnist digits from train dataset
    indexes = np.random.randint(0, x_train.shape[0], size=25)
    images = x_train[indexes]
    labels = y_train[indexes]

    #plot the 25 minist digits
    plt.figure(figsize=(5,5))
    for i in range(len(indexes)):
        plt.subplot(5, 5, i + 1)
        image = images[i]
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.savefig("mnist-samples.png")
    plt.show()
    plt.close('all')

dataset_checkout()