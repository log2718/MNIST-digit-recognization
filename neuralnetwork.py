import numpy as np
import random

'''
Simoid activation function, NumPy applies it elementwise
'''
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

'''
Returns the derivative of the sigmoid function for backprop
'''
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

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
    
    '''
    Updates the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
    mini_batch: list of (x,y) tuples
    eta: learning rate
    '''
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) # gradient for the single training example
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    '''
    Returns the number of correct results outputted by the Neural Network given the test data.
    Uses argmax to find the output with the highest activation and compares it to the correct answer.
    '''
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)

    '''
    Returns the gradient for the cost function C_x.
    The gradient is a tuple (nabla_b, nabla_w), which are lists of numpy arrays
    '''
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store activations
        zs = [] # list to store z vectors
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    '''
    Returns the vector of partial derivatives partial C_x / partial a for the output activations.
    '''
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    

    



