import numpy as np

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

    
