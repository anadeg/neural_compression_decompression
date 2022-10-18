import numpy as np
from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, mode, outfilename):
    # npdata = np.asarray(npdata, dtype=np.float32)
    img = Image.fromarray(npdata.astype('uint8'), mode=mode)
    # img.save(outfilename)
    img.show()


class NeuralNetworkNoBias:
    def __init__(self, X, hidden_layers=100):
        self.coefficient = (1/1)*10**1
        self.h, self.w, self.dim = X.shape
        self.size = self.h * self.w
        self.hidden_layers = hidden_layers
        self.X = X.reshape(self.size, self.dim) / 255
        self.Y = self.X
        self.W1 = self.he_initialization(self.size, hidden_layers, self.size)
        self.W2 = self.he_initialization(self.hidden_layers, self.size, hidden_layers)

    @staticmethod
    def he_initialization(l_1_size, rows, cols):
        return np.random.randn(rows, cols) * np.sqrt(2/l_1_size)

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
    def parametric_relu(Z):
        return np.where(Z >= 0, 0.01*Z, 0.001*Z)

    @staticmethod
    def parametric_relu_derivative(Z):
        return np.where(Z >= 0, 0.01, -0.001)

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_derivative(Z):
        return NeuralNetworkNoBias.sigmoid(Z) * (1 - NeuralNetworkNoBias.sigmoid(Z))

    @staticmethod
    def tanh(Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    @staticmethod
    def tanh_derivative(Z):
        return 1 - np.power(NeuralNetworkNoBias.tanh(Z), 2)

    @staticmethod
    def stable_softmax(Z):
        x = Z - np.max(Z, axis=-1, keepdims=True)
        numerator = np.exp(x)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        return np.exp(x) / denominator

    @staticmethod
    def stable_softmax_derivative(Z):
        pass

    def gradient_descent(self, learning_rate, iterations):
        for i in range(1, iterations+1):
            if i % 5 == 0:
                print(f'{i}-th iteration')
            Y_observed = self.Y

            A1 = self.W1 @ self.X
            A2 = self.W2 @ A1

            dx = A2 - Y_observed
            dW2 = dx @ A1.T
            dW1 = self.W2.T @ dx @ self.X.T

            self.W1 -= learning_rate * dW1
            self.W2 -= learning_rate * dW2

    def fit(self, learning_rate=2*10**(-4), iterations=50):
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)

    def predict(self, X):
        X = X.reshape(self.size, self.dim) / 255
        A1 = self.W1 * X
        A2 = self.W2 * A1
        # A2 = np.where(A2 >= 0, A2, 0)

        A2 = A2.reshape(self.h, self.w, self.dim)
        A2 = np.round(255 * A2)
        # save_image(A2, "RGB", 's')
        return A2


if __name__ == '__main__':
    data = load_image('../imgs/flower.jpg')

    simple_nn = NeuralNetworkNoBias(data, hidden_layers=30*20)
    simple_nn.fit()
    output = simple_nn.predict(data)
    print(data[:5, :5])
    print(output[:5, :5])

    save_image(output, "RGB", '../imgs/kitten_output.bmp')