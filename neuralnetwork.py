import numpy as np
import random

'''
Simoid activation function, NumPy applies it elementwise
'''
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

'''
Network class to represent a neural network
sizes: the number of neurons in the respective layers
'''
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #to be changed with LeCun initialization later
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        # each value in the weights list is w_jk, 
        # indicating the weight from the kth neuron in the previous layer 
        # to the jth neuron in the current layer

    '''
    Returns the output of the network
    a: input to the network
    '''
    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    '''
    Trains the Network using mini-batch stochastic gradient descent.
    training_data: list of (x,y) tuples to train on
    epochs, mini_batch_size, eta: hyperparameters
    test_data: Optional test set to evaluate on after each epoch
    '''
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete.")
    
    def update_mini_batch(self, mini_batch, eta):
        pass

    def evaluate(self, test_data):
        pass

    def backprop(self, x, y):
        pass
    

    



