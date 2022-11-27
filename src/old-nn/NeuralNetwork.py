import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate, epochs, loss_function, loss_prime):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []
        self._loss_function = loss_function
        self._loss_prime = loss_prime

    def add(self, layer):
        self.layers.append(layer)

    def propagate_through_layers(self, input_vector):
        output_vector = input_vector
        for layer in self.layers:
            output_vector = layer.forward_propagation(output_vector)
        return output_vector

    def initiate_weights_of_input_layer(self, n_neurons):
        input_layer = self.layers[0]  # get first layer
        input_layer.input_size = n_neurons
        input_layer.weights = input_layer.initiate_weights(input_layer.input_size, input_layer.output_size, (-1, 1))

    def fit(self, x_train, y_train, verbose=True):
        error = 0
        for epoch in range(1, self.epochs+1):
            classified_correctly_counter = 0
            for x, y in zip(x_train, y_train):
                output = self.propagate_through_layers(x)

                # calculate error
                error = self._loss_function(y, output)

                classified_correctly_counter += int(np.argmax(output) == np.argmax(y))

                # calculate loss function
                grad = self._loss_prime(y, output)

                # propagate backward
                for layer in reversed(self.layers):
                    grad = layer.backward_propagation(grad, self.learning_rate)

            error /= len(x_train)

            if verbose:
                cc_rate = round(classified_correctly_counter*100/len(y_train), ndigits=4)
                print(f"Epoch {epoch}, classified correctly: {cc_rate}, error {error}")
