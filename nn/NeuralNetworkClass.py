import numpy as np
from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    # npdata = np.asarray(npdata, dtype=np.float32)
    img = Image.fromarray(npdata.astype('uint8'), mode="RGB")
    # img.save(outfilename)
    img.show()


class NeuralNetwork:
    def __init__(self, X, hidden_layers=100):
        self.coefficient = (1/1)*10**1
        self.h, self.w, self.dim = X.shape
        self.size = self.h * self.w
        self.hidden_layers = hidden_layers
        self.X = X.reshape(self.size, self.dim) / 255
        self.Y = self.X
        self.W1 = (np.random.random((hidden_layers, self.size)) - 0.5) / self.coefficient
        self.W2 = (np.random.random((self.size, hidden_layers)) - 0.5) / self.coefficient
        self.b1 = np.zeros((hidden_layers, 1)) # / self.coefficient
        self.b2 = np.zeros((self.size, 1)) # / self.coefficient

    @staticmethod
    def relu(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def relu_derivative(Z):
        return Z > 0

    @staticmethod
    def leaky_relu(Z):
        return np.maximum(0.2*Z, Z)

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
        # A1 = self.sigmoid(Z1)
        # A1 = self.tanh(Z1)
        # A1 = self.parametric_relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        # A2 = self.stable_softmax(Z2)
        A2 = self.sigmoid(Z2)
        # A2 = self.tanh(Z2)
        # A2 = self.leaky_relu(Z2)

        return Z1, Z2, A1, A2

    def backward_propagation(self, Z1, Z2, A1, A2):
        Y_observed = self.Y

        dA2 = A2 - Y_observed
        dZ2 = dA2 * self.sigmoid_derivative(Z2)
        # dZ2 = self.leaky_relu_derivative(Z2)
        # dZ2 = dA2 * self.tanh_derivative(Z2)
        # dZ2 = A2 - Y_observed
        dW2 = np.dot(dZ2, A1.T)     # (60000, 3) dot (3, 1800)
        dB2 = 1 / dZ2.shape[1] * np.sum(dZ2, axis=1).reshape(-1, 1)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.leaky_relu_derivative(Z1)
        # dZ1 = dA1 * self.sigmoid_derivative(Z1)
        # dZ1 = dA1 * self.tanh_derivative(Z1)
        # dZ1 = dA1 * self.parametric_relu_derivative(Z1)
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

    def gradient_descent(self, learning_rate=0.001, iterations=30):
        for i in range(1, iterations+1):
            if i % 5 == 0:
                print(f'{i}-th iteration')
            Z1, Z2, A1, A2 = self.forward_propagation(self.X)
            dW1, dW2, dB1, dB2 = self.backward_propagation(Z1, Z2, A1, A2)
            self.update_parameters(dW1, dW2, dB1, dB2, learning_rate=learning_rate)

    def fit(self, learning_rate=1*10**(-4), iterations=75):
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)

    def predict(self, X):
        X = X.reshape(self.size, self.dim) / 255
        _, _, _, A2 = self.forward_propagation(X)
        # A2 = np.where(A2 >= 0, A2, 0)
        A2 = A2.reshape(self.h, self.w, self.dim)
        A2 = np.round(255 * A2)
        return A2


if __name__ == '__main__':
    # data = np.array([[[23, 45, 109], [0, 23, 23], [1, 0, 0]],
    #                  [[56, 89, 10], [23, 76, 87], [100, 10, 0]],
    #                  [[9, 7, 199], [23, 65, 63], [34, 45, 67]]])
    data = load_image('../imgs/kitten.jpg')
    # data = np.random.randint(255, size=(300, 200, 3))
    # print(data, '\n\n=====================\n')
    simple_nn = NeuralNetwork(data, hidden_layers=20*12*3)
    simple_nn.fit()
    output = simple_nn.predict(data)
    print(data[:5, :5])
    print(output[:5, :5])
    # # output = np.where(output >= 0, output, 0)
    # # output = output.reshape(3, 3, 3).round()
    # save_image(data, 'ffkfk')
    save_image(output, '../imgs/kitten_output.bmp')



















