#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np


class DeepNeuralNetwork:
    """ Class representing a deep neural network """
    def __init__(self, nx, layers):
        """ Init
            nx is the number of input features
                nx must be a positive integer
            layers is a list representing the number of nodes in each layer
                layers must be a list of positive integers
            L: number of layers in the network
            cache: dictionary to hold all intermediary values of the network
            weights: dictionary to hold all weights and biases of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(0, self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights['W1'] = w
                self.__weights['b1'] = np.zeros((layers[i], 1))
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w *= np.sqrt(2 / layers[i - 1])
                self.__weights['W{}'.format(i + 1)] = w
                self.__weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ L getter """
        return self.__L

    @property
    def cache(self):
        """ cache getter """
        return self.__cache

    @property
    def weights(self):
        """ weights getter """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network
            X with shape (nx, m) contains input data
                nx is the number of input features
                m is the number of examples
            Updates __cache
        """
        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            w = "W" + str(i)
            a = "A" + str(i - 1)
            b = "b" + str(i)
            Y = self.weights[w] @ self.cache[a] + self.weights[b]
            A = 1 / (1 + np.exp(Y * -1))
            self.__cache["A" + str(i)] = A

        return self.__cache["A" + str(self.L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression
            Y is a numpy.ndarray with shape (1, m) that contains the correct
                labels for the input data
            A is a numpy.ndarray with shape (1, m) containing the activated
                output of the neuron for each example
            Returns calculated cost
        """
        # Using L(Y, A) = -(Ylog(A) + (1 - Y)log(1 - A)) loss function
        cost = np.matmul(Y, np.log(A).T)
        cost += np.matmul((1 - Y), np.log(1.0000001 - A).T)
        cost /= len(Y[0]) * -1
        return cost[0][0]

    def evaluate(self, X, Y):
        """ Evaluates the neuron's predictions
            X is a numpy.ndarray with shape(nx, m) that contains input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y is a numpy.ndarray with shape(1, m) that contains the correct
                labels for the input data
            Returns the neuron's prediction and the cost of the network
                The prediction is a numpy.ndarray with shape (1, m) containing
                the predicted labels for each example
                The label values are 1 if the output of the network is >= 0.5
                    and 0 otherwise
        """
        self.forward_prop(X)
        A = self.cache["A" + str(self.L)]
        P = np.array([list(map(lambda x: 1 if x >= 0.5 else 0, A[0]))])
        return P, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network
            Y with shape (1, m) contains the correct labels for input data
            cache is a dictionary containing all the intermediary values of
                the network
            alpha is the learning rate
            Updates __weights
        """
        m = len(Y[0])
        dz2 = cache["A" + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            A = cache["A" + str(i - 1)]
            W = self.weights["W" + str(i)]
            dz1 = (W.T @ dz2) * (A * (1 - A))
            dw = dz2 @ A.T / m
            db = np.sum(dz2, axis=1, keepdims=True) / m
            dz2 = dz1
            self.__weights["W" + str(i)] -= alpha * dw
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y,
              iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains the deep neural network
            X with shape (nx, m) contains input data
                nx is the number of input features
                m is the number of examples
            Y with shape (1, m) contains the correct labels for input data
            iterations is the number of iterations to train over
                iterations must be a positive integer
            alpha is the learning rate
                alpha must be a positive float
            Updates __weights and __cache
            Returns evaluation of the training data after all iterations
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be a positive float")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        import matplotlib.pyplot as plt
        x = []
        y = []
        for i in range(0, iterations):
            self.forward_prop(X)

            if verbose:
                if i == iterations - 1:
                    cost = self.cost(Y, self.cache["A" + str(self.L)])
                    print("Cost after {} iterations: {}".format(i + 1, cost))
                elif i % step == 0:
                    cost = self.cost(Y, self.cache["A" + str(self.L)])
                    print("Cost after {} iterations: {}".format(i, cost))

            if graph:
                if i == iterations - 1:
                    x.append(i + 1)
                    y.append(self.cost(Y, self.cache["A" + str(self.L)]))
                elif i % step == 0:
                    x.append(i)
                    y.append(self.cost(Y, self.cache["A" + str(self.L)]))

            self.gradient_descent(Y, self.cache, alpha)

        if graph:
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
