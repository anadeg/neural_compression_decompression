import numpy as np


class NeuralNetwork:
    """
    X = [[12, 0, 17], [23, 45, 0], [79, 93, 0], [66, 7, 7]]
    """
    def __init__(self, X, hidden_layer):
        size, dim = X.shape
        self.X = X
        self.W1 = np.random.rand(size, dim)
        self.b1 = np.zeros(hidden_layer, dim)
        self.W2 = np.random.rand(hidden_layer, size)
        self.b2 = np.zeros(hidden_layer, size)

    @staticmethod
    def relu(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def relu_derivative(Z):
        return Z > 0

    @staticmethod
    def leaky_relu(Z):
        return np.maximum(0.1*Z, Z)

    @staticmethod
    def leaky_relu_derivative(Z):
        return np.where(Z >= 0, 1, 0.1)

    @staticmethod
    def softmax(Z):
        return np.exp(Z) / sum(np.exp(Z))

    def forward_propagation(self, X):
        Z1 = np.dot(self.X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        # A2 = self.softmax(Z2) # idiot
        A2 = self.relu(Z2)
        return Z1, Z2, A1, A2

    def back_propagation(self, Z1, Z2, A1, A2, Y_observed):
        # SSR = np.power(Z2 - Y_predicted, 2)
        # d SSR / d A2 = 2 * (Y_observed - Y_predicted)
        # d A2 / d Z2 = d (activation function) / d Z2
        # d Z2 / d W2 = A1
        # d SSR / d W2 = 1/ n * sum (d SSR / d W2)

        # d SSR / d B2 = 1/ n * sum(d (act. func) * 2*(Y_obs-Y_pred))

        # d SSR / d A1 = 1 / n * sum(W2 * d (act.func) * 2*(Y_obs-Y_pred))

        Y_observed = Y_observed.T

        dSSR_dA2 = -2 * (A2 - Y_observed)
        dA2_dZ2 = self.relu_derivative(Z2)
        dZ2_dW2 = A1
        dZ2_dB2 = 1
        dZ2_dA1 = self.W2

        dA1_dZ1 = self.relu_derivative(Z1)
        dZ1_dW1 = self.X
        dZ1_dB1 = 1
        dZ1_dA0 = self.W1

        dW2 = 1/ np.sum(dSSR_dA2 * dA2_dZ2 * dZ2_dW2)
        dB2 = dSSR_dA2 * dA2_dZ2 * dZ2_dB2
        dA1 = dSSR_dA2 * dA2_dZ2 * dZ2_dA1

        dW1 = dA1 * dA1_dZ1 * dZ1_dW1
        dB1 = dA1 * dA1_dZ1 * dZ1_dB1
        dA0 = dA1 * dA1_dZ1 * dZ1_dA0










