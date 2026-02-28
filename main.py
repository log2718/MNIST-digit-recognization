import mnistloader
import neuralnetwork

training_data, validation_data, test_data = mnistloader.load_data_wrapper()
net = neuralnetwork.Network([784, 30, 10])

#train the NN
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

