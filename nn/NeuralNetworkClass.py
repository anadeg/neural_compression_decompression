import numpy as np


class NeuralNetwork:
    def __init__(self, X, hidden_layers=100):
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        size, dim = X.shape
        self.X = X.T / 255
        self.Y = self.X
        self.W1 = np.random.rand(size, hidden_layers)
        self.W2 = np.random.rand(hidden_layers, size)
        self.b1 = np.zeros((dim, hidden_layers))
        self.b2 = np.zeros((dim, size))

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
        print('f p')
        Z1 = np.dot(self.X, self.W1) + self.b1
        A1 = self.relu(Z1)
        # A1 is compressed image
        # download it
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.relu(Z2)
        return Z1, Z2, A1, A2

    def backward_propagation(self, Z1, Z2, A1, A2):
        # SSR = np.power(Z2 - Y_predicted, 2)
        # d SSR / d A2 = 2 * (Y_observed - Y_predicted)
        # d A2 / d Z2 = d (activation function) / d Z2
        # d Z2 / d W2 = A1
        # d SSR / d W2 = 1/ n * sum (d SSR / d W2)

        # d SSR / d B2 = 1/ n * sum(d (act. func) * 2*(Y_obs-Y_pred))

        # d SSR / d A1 = 1 / n * sum(W2 * d (act.func) * 2*(Y_obs-Y_pred))
        print('f p')
        Y_observed = self.Y

        dSSR_dA2 = (1 / Y_observed.size) * np.sum((-2 * (A2 - Y_observed)))
        dA2_dZ2 = (1 / Z2.size) * np.sum(self.relu_derivative(Z2))
        dZ2_dW2 = (1 / A1.size) * np.sum(A1)
        dZ2_dB2 = 1
        dZ2_dA1 = (1 / self.W2.size) * np.sum(self.W2)

        dA1_dZ1 = (1 / Z1.size) * np.sum(self.relu_derivative(Z1))
        dZ1_dW1 = (1 / self.X.size) * np.sum(self.X)
        dZ1_dB1 = 1
        dZ1_dA0 = (1 / self.W1.size) * np.sum(self.W1)

        # does it work?
        dW2 = dSSR_dA2 * dA2_dZ2 * dZ2_dW2
        dB2 = dSSR_dA2 * dA2_dZ2 * dZ2_dB2
        dA1 = dSSR_dA2 * dA2_dZ2 * dZ2_dA1

        dW1 = dA1 * dA1_dZ1 * dZ1_dW1
        dB1 = dA1 * dA1_dZ1 * dZ1_dB1
        # dA0 = 1/n * np.sum(dA1 * dA1_dZ1 * dZ1_dA0)

        return dW1, dW2, dB1, dB2

    def update_parameters(self, dW1, dW2, dB1, dB2, learning_rate=0.05):
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2
        self.b1 -= learning_rate * dB1
        self.b2 -= learning_rate * dB2

    def gradient_descent(self, X, learning_rate=0.01, iterations=50):
        for i in range(iterations):
            Z1, Z2, A1, A2 = self.forward_propagation()
            dW1, dW2, dB1, dB2 = self.backward_propagation(Z1, Z2, A1, A2)
            self.update_parameters(dW1, dW2, dB1, dB2, learning_rate=learning_rate)

    def fit(self, learning_rate=0.05, iterations=500):
        self.gradient_descent(self.X, learning_rate=learning_rate, iterations=iterations)

    def predict(self):
        _, _, _, A2 = self.forward_propagation()
        return A2.T * 255


if __name__ == '__main__':
    data = np.array([[[23, 45, 109], [0, 23, 23], [1, 0, 0]],
                     [[56, 89, 10], [23, 76, 87], [100, 10, 0]],
                     [[9, 7, 199], [23, 65, 63], [34, 45, 67]]])
    print(data.shape)
    simple_nn = NeuralNetwork(data, hidden_layers=4)
    simple_nn.fit()
    output = simple_nn.predict()
    print(output)



















