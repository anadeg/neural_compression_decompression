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


class NeuralNetwork:
    def __init__(self, X, hidden_layers=100):
        self.coefficient = (1/1)*10**1
        self.h, self.w, self.dim = X.shape
        self.size = self.h * self.w
        self.hidden_layers = hidden_layers
        # change x
        # self.X = X.reshape(self.size, self.dim) / 255
        # self.X = X.reshape(self.size, self.dim)
        self.X = self.compress(X)

        self.new_h, self.new_w, d = self.X.shape
        self.new_size = self.new_h * self.new_w

        self.X = self.X.reshape(self.new_size, self.dim) / 255
        self.Y = self.X

        self.W1 = self.he_initialization(self.new_size, hidden_layers, self.new_size) / self.coefficient
        self.W2 = self.he_initialization(self.hidden_layers, self.new_size, hidden_layers) / self.coefficient
        self.b1 = np.zeros((hidden_layers, 1))
        self.b2 = np.zeros((self.new_size, 1))

    def compress(self, x):
        self.c_coeff = 4
        new_h, new_w = self.h // self.c_coeff, self.w // self.c_coeff
        new_x = np.zeros((new_h, new_w, self.dim))
        for i in range(new_h):
            for j in range(new_w):
                for d in range(self.dim):
                    start_i = self.c_coeff * i
                    start_j = self.c_coeff * j
                    temp = x[start_i: start_i + self.c_coeff, start_j: start_j + self.c_coeff, d]
                    value = temp.mean()
                    new_x[i, j, d] = value
        return new_x

    @staticmethod
    def he_initialization(l_1_size, rows, cols):
        # print(size)
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
        return NeuralNetwork.sigmoid(Z) * (1 - NeuralNetwork.sigmoid(Z))

    @staticmethod
    def tanh(Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    @staticmethod
    def tanh_derivative(Z):
        return 1 - np.power(NeuralNetwork.tanh(Z), 2)

    @staticmethod
    def stable_softmax(Z):
        x = Z - np.max(Z, axis=-1, keepdims=True)
        numerator = np.exp(x)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        # denominator = denominator.reshape(denominator.size, 1)
        return np.exp(x) / denominator

    @staticmethod
    def stable_softmax_derivative(Z):
        pass

    def forward_propagation(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.leaky_relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = np.copy(Z2)

        return Z1, Z2, A1, A2

    def backward_propagation(self, Z1, Z2, A1, A2):
        Y_observed = self.Y

        dA2 = A2 - Y_observed
        dZ2 = dA2.copy()
        dW2 = np.dot(dZ2, A1.T)
        dB2 = 1 / dZ2.shape[1] * np.sum(dZ2, axis=1).reshape(-1, 1)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.leaky_relu_derivative(Z1)
        dW1 = np.dot(dZ1, self.X.T)
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

    def gradient_descent(self, learning_rate=1*10**(-4), iterations=30):
        for i in range(1, iterations+1):
            if i % 5 == 0:
                print(f'{i}-th iteration')
            Z1, Z2, A1, A2 = self.forward_propagation(self.X)
            dW1, dW2, dB1, dB2 = self.backward_propagation(Z1, Z2, A1, A2)
            self.update_parameters(dW1, dW2, dB1, dB2, learning_rate=learning_rate)

    def fit(self, learning_rate=1*10**(-5), iterations=200):
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)

    def predict(self, X):
        _, _, A1, A2 = self.forward_propagation(X)

        A2 = A2.reshape(self.new_h, self.new_w, self.dim)
        A2 = np.round(255 * A2)
        A2 = np.where(A2 > 255, 255, A2)
        A2 = np.where(A2 < 0, 0, A2)
        return A2


if __name__ == '__main__':
    data = load_image('../imgs/kitten.jpg')

    simple_nn = NeuralNetwork(data, hidden_layers=80*40)
    simple_nn.fit()
    output = simple_nn.predict(simple_nn.X)
    print(data[:5, :5])
    print(output[:5, :5])

    save_image(output, "RGB", '')

    # data_test = np.array([[[1, 1], [2, 1], [3, 1], [4, 1], [2, 2], [3, 2], [4, 2], [5, 2]],
    #                       [[2, 1], [3, 1], [4, 1], [1, 1], [1, 2], [1, 2], [0, 2], [1, 2]],
    #                       [[3, 1], [3, 1], [1, 1], [1, 1], [0, 2], [0, 2], [1, 2], [3, 2]],
    #                       [[8, 1], [5, 1], [6, 1], [4, 1], [3, 2], [2, 2], [2, 2], [1, 2]]])
    # test_nn = NeuralNetwork(data, hidden_layers=10)
    # # print(data_test[[2, 2, 0], [0, 4, 0]])
    # n_d_test = test_nn.compress()
    # # print(n_d_test[20: 25, 20: 25, :])



















