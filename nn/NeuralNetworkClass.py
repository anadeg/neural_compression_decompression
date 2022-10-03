import numpy as np


class NeuralNetwork:
    """
    X = [[12, 0, 17], [23, 45, 0], [79, 93, 0], [66, 7, 7]]
    """
    def __init__(self, X, Y, hidden_layer):
        size, dim = X.shape
        self.X = X
        self.Y = Y
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

    def forward_propagation(self):
        Z1 = np.dot(self.X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        # A2 = self.softmax(Z2) # idiot
        A2 = self.relu(Z2)
        return Z1, Z2, A1, A2

    def back_propagation(self, Z1, Z2, A1, A2):
        # SSR = np.power(Z2 - Y_predicted, 2)
        # d SSR / d A2 = 2 * (Y_observed - Y_predicted)
        # d A2 / d Z2 = d (activation function) / d Z2
        # d Z2 / d W2 = A1
        # d SSR / d W2 = 1/ n * sum (d SSR / d W2)

        # d SSR / d B2 = 1/ n * sum(d (act. func) * 2*(Y_obs-Y_pred))

        # d SSR / d A1 = 1 / n * sum(W2 * d (act.func) * 2*(Y_obs-Y_pred))

        Y_observed = self.Y.T

        dSSR_dA2 = -2 * (A2 - Y_observed)
        dA2_dZ2 = self.relu_derivative(Z2)
        dZ2_dW2 = A1
        dZ2_dB2 = 1
        dZ2_dA1 = self.W2

        dA1_dZ1 = self.relu_derivative(Z1)
        dZ1_dW1 = self.X
        dZ1_dB1 = 1
        dZ1_dA0 = self.W1

        n = Y_observed.size     # does it work?
        dW2 = 1/n * np.sum(dSSR_dA2 * dA2_dZ2 * dZ2_dW2)
        dB2 = 1/n * np.sum(dSSR_dA2 * dA2_dZ2 * dZ2_dB2)
        dA1 = 1/n * np.sum(dSSR_dA2 * dA2_dZ2 * dZ2_dA1)

        dW1 = 1/n * np.sum(dA1 * dA1_dZ1 * dZ1_dW1)
        dB1 = 1/n * np.sum(dA1 * dA1_dZ1 * dZ1_dB1)
        # dA0 = 1/n * np.sum(dA1 * dA1_dZ1 * dZ1_dA0)

        return dW1, dW2, dB1, dB2

    def update_parameters(self, dW1, dW2, dB1, dB2, learning_rate=0.05):
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2
        self.b1 -= learning_rate * dB1
        self.b2 -= learning_rate * dB2

    def gradient_descent(self, learning_rate=0.05, iterations=500):
        for i in range(iterations):
            Z1, Z2, A1, A2 = self.forward_propagation()
            dW1, dW2, dB1, dB2 = self.back_propagation(Z1, Z2, A1, A2)
            self.update_parameters(dW1, dW2, dB1, dB2, learning_rate=learning_rate)

    def train(self, learning_rate=0.05, iterations=500):
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)















