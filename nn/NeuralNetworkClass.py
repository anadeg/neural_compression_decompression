import numpy as np
from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


class NeuralNetwork:
    def __init__(self, X, hidden_layers=100):
        self.h, self.w, self.dim = X.shape
        self.size = self.h * self.w * self.dim
        self.hidden_layers = hidden_layers
        X = X.reshape(1, X.size)
        self.X = X / 255
        self.Y = self.X
        self.W1 = np.random.rand(self.size, hidden_layers) / (10**4)
        self.W2 = np.random.rand(hidden_layers, self.size) / (10**4)
        self.b1 = np.zeros((1, hidden_layers))
        self.b2 = np.zeros((1, self.size))

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
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.leaky_relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2

        return Z1, Z2, A1

    def backward_propagation(self, Z1, Z2, A1):
        Y_observed = self.Y

        dZ2 = (Z2 - Y_observed)
        dW2 = 1 / self.size * np.dot(A1.T, dZ2)
        dB2 = 1/ self.size * np.sum(dZ2, axis=0)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.leaky_relu_derivative(Z1)
        dW1 = 1 / self.hidden_layers * np.dot(self.X.T, dZ1)
        dB1 = 1 / self.hidden_layers * np.sum(dZ1, axis=0)

        return dW1, dW2, dB1, dB2

    def update_parameters(self, dW1, dW2, dB1, dB2, learning_rate=0.01):
        self.W1 = self.W1 - learning_rate * dW1
        self.W2 = self.W2 - learning_rate * dW2
        self.b1 = self.b1 - learning_rate * dB1
        self.b2 = self.b2 - learning_rate * dB2

    def gradient_descent(self, learning_rate=0.001, iterations=100):
        for i in range(iterations):
            Z1, Z2, A1 = self.forward_propagation(self.X)
            dW1, dW2, dB1, dB2 = self.backward_propagation(Z1, Z2, A1)
            self.update_parameters(dW1, dW2, dB1, dB2, learning_rate=learning_rate)

    def fit(self, learning_rate=0.01, iterations=100):
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)

    def predict(self, X):
        X = X.reshape(1, -1) / 255
        _, Z2, _ = self.forward_propagation(X)
        Z2 = np.where(Z2 >= 0, Z2, 0)
        Z2 = Z2.reshape(self.h, self.w, self.dim)
        Z2 = np.round(Z2 * 255)
        return Z2


if __name__ == '__main__':
    # data = np.array([[[23, 45, 109], [0, 23, 23], [1, 0, 0]],
    #                  [[56, 89, 10], [23, 76, 87], [100, 10, 0]],
    #                  [[9, 7, 199], [23, 65, 63], [34, 45, 67]]])
    # data = load_image('../imgs/sample_2.bmp')
    data = np.random.randint(255, size=(300, 200, 3))
    # print(data, '\n\n=====================\n')
    simple_nn = NeuralNetwork(data, hidden_layers=30*20*3)
    simple_nn.fit()
    output = simple_nn.predict(data)
    print(output - data)
    # # output = np.where(output >= 0, output, 0)
    # # output = output.reshape(3, 3, 3).round()
    # save_image(output, '../imgs/sample_2_output.bmp')



















