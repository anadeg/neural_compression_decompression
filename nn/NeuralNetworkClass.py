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
        self.coefficient = 10**4
        self.h, self.w, self.dim = X.shape
        self.size = self.h * self.w
        self.hidden_layers = hidden_layers
        self.X = X.reshape(self.size, self.dim)
        self.Y = self.X
        self.W1 = np.random.rand(hidden_layers, self.size) / self.coefficient
        self.W2 = np.random.rand(self.size, hidden_layers) / self.coefficient
        self.b1 = np.zeros((hidden_layers, 1)) / self.coefficient
        self.b2 = np.zeros((self.size, 1)) / self.coefficient

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
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.leaky_relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2

        return Z1, Z2, A1

    def backward_propagation(self, Z1, Z2, A1):
        Y_observed = self.Y

        dZ2 = (Z2 - Y_observed)
        dW2 = np.dot(dZ2, A1.T)     # (60000, 3) dot (3, 1800)
        dB2 = 1 / dZ2.shape[1] * np.sum(dZ2, axis=1).reshape(-1, 1)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.leaky_relu_derivative(Z1)
        dW1 = np.dot(dZ1, self.X.T)  # (1800, 3) dot (60000, 3).T
        dB1 = 1 / dZ1.shape[1] * np.sum(dZ1, axis=1).reshape(-1, 1)

        return dW1, dW2, dB1, dB2

    def update_parameters(self, dW1, dW2, dB1, dB2, learning_rate=0.01):
        tempW1 = dW1 * learning_rate
        tempW2 = dW2 * learning_rate
        tempB1 = dB1 * learning_rate
        tempB2 = dB2 * learning_rate
        self.W1 = self.W1 - tempW1
        self.W2 = self.W2 - tempW2
        self.b1 = self.b1 - tempB1
        self.b2 = self.b2 - tempB2

    def gradient_descent(self, learning_rate=0.001, iterations=100):
        for i in range(1, iterations+1):
            if i % 5 == 0:
                print(f'{i}-th iteration')
            Z1, Z2, A1 = self.forward_propagation(self.X)
            dW1, dW2, dB1, dB2 = self.backward_propagation(Z1, Z2, A1)
            self.update_parameters(dW1, dW2, dB1, dB2, learning_rate=learning_rate)

    def fit(self, learning_rate=10**(-12), iterations=100):
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)

    def predict(self, X):
        X = X.reshape(self.size, self.dim)
        _, Z2, _ = self.forward_propagation(X)
        Z2 = np.where(Z2 >= 0, Z2, 0)
        Z2 = Z2.reshape(self.h, self.w, self.dim)
        Z2 = np.round(Z2)
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
    print(data[:5, :5])
    print(output[:5, :5])
    # # output = np.where(output >= 0, output, 0)
    # # output = output.reshape(3, 3, 3).round()
    # save_image(output, '../imgs/sample_2_output.bmp')



















