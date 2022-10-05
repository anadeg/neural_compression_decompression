import numpy as np


class NeuralNetwork:
    def __init__(self, X, hidden_layers=100):
        self.h, self.w, self.dim = X.shape
        size = self.h * self.w * self.dim
        X = X.reshape(1, X.size)
        self.X = X / 255
        self.Y = self.X
        self.W1 = np.random.rand(size, hidden_layers)
        self.W2 = np.random.rand(hidden_layers, size)
        self.b1 = np.zeros((1, hidden_layers))
        self.b2 = np.zeros((1, size))

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
        # print('f p')
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.leaky_relu(Z1)
        # A1 is compressed image
        # download it
        Z2 = np.dot(A1, self.W2) + self.b2
        return Z1, Z2, A1

    def backward_propagation(self, Z1, Z2, A1):
        # SSR = np.power(Z2 - Y_predicted, 2)
        # d SSR / d A2 = 2 * (Y_observed - Y_predicted)
        # d A2 / d Z2 = d (activation function) / d Z2
        # d Z2 / d W2 = A1
        # d SSR / d W2 = 1/ n * sum (d SSR / d W2)

        # d SSR / d B2 = 1/ n * sum(d (act. func) * 2*(Y_obs-Y_pred))

        # d SSR / d A1 = 1 / n * sum(W2 * d (act.func) * 2*(Y_obs-Y_pred))
        # print('f p')
        Y_observed = self.Y

        # SSE = np.sum(np.power(A2 - Y_observed), 2)

        dZ2 = -2 * (Y_observed - Z2)
        # print(dZ2.shape)
        dW2 = 1 / dZ2.size * np.dot(A1.T, dZ2)
        dB2 = 1 / dZ2.size * np.sum(dZ2)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.leaky_relu_derivative(Z1)
        dW1 = 1 / dZ1.size * np.dot(self.X.T, dZ1)
        dB1 = 1 / dZ1.size * np.sum(dZ1)

        return dW1, dW2, dB1, dB2

    def update_parameters(self, dW1, dW2, dB1, dB2, learning_rate=0.01):
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2
        self.b1 -= learning_rate * dB1
        self.b2 -= learning_rate * dB2

    def gradient_descent(self, learning_rate=0.01, iterations=100):
        for i in range(iterations):
            Z1, Z2, A1 = self.forward_propagation(self.X)
            dW1, dW2, dB1, dB2 = self.backward_propagation(Z1, Z2, A1)
            self.update_parameters(dW1, dW2, dB1, dB2, learning_rate=learning_rate)

    def fit(self, learning_rate=0.05, iterations=200):
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)

    def predict(self, X):
        X = X.reshape(1, -1) / 255
        _, Z2, _ = self.forward_propagation(X)
        return Z2 * 255


if __name__ == '__main__':
    data = np.array([[[23, 45, 109], [0, 23, 23], [1, 0, 0]],
                     [[56, 89, 10], [23, 76, 87], [100, 10, 0]],
                     [[9, 7, 199], [23, 65, 63], [34, 45, 67]]])
    # print(data.shape)
    simple_nn = NeuralNetwork(data, hidden_layers=9)
    simple_nn.fit()
    output = simple_nn.predict(data)
    output = output.reshape(3, 3, 3).round()
    print(output)



















