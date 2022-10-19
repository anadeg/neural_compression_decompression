import numpy as np
from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, mode, outfilename):
    img = Image.fromarray(npdata.astype('uint8'), mode=mode)
    # img.save(outfilename)
    img.show()


class NeuralNetwork:
    def __init__(self, X, c_coeff=2, hidden_layers=(30, 20)):
        self.coefficient = (1/1)*10**1
        self.h, self.w, self.dim = X.shape
        self.size = self.h * self.w
        self.hidden_h, self.hidden_w = hidden_layers
        self.hidden_layers = self.hidden_h * self.hidden_w
        self.c_coeff = c_coeff

        self.X = self.compress(X)

        self.new_h, self.new_w, d = self.X.shape
        self.new_size = self.new_h * self.new_w

        self.X = self.X.reshape(self.new_size, self.dim) / 255
        self.Y = self.X

        self.W1 = self.he_initialization(self.new_size, self.hidden_layers, self.new_size) / self.coefficient
        self.W2 = self.he_initialization(self.hidden_layers, self.new_size, self.hidden_layers) / self.coefficient
        self.b1 = np.zeros((self.hidden_layers, 1))
        self.b2 = np.zeros((self.new_size, 1))

    def compress(self, x):
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

    def decompress(self, new_x):
        back = np.zeros((self.h, self.w, self.dim))
        for i in range(len(new_x)):
            for j in range(len(new_x[i])):
                for d in range(self.dim):
                    start_i = self.c_coeff * i
                    start_j = self.c_coeff * j
                    back[start_i: start_i + self.c_coeff, start_j: start_j + self.c_coeff, d] = new_x[i][j][d]
        return back




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

    def forward_propagation(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
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
        dZ1 = dA1 * self.relu_derivative(Z1)
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

    def gradient_descent(self, learning_rate=1*10**(-4), iterations=100):
        for i in range(1, iterations+1):
            if i % 5 == 0:
                print(f'{i}-th iteration')
            Z1, Z2, A1, A2 = self.forward_propagation(self.X)
            dW1, dW2, dB1, dB2 = self.backward_propagation(Z1, Z2, A1, A2)
            self.update_parameters(dW1, dW2, dB1, dB2, learning_rate=learning_rate)

    def fit(self, learning_rate=1*10**(-5), iterations=100):
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)

    def predict(self, X):
        _, _, A1, A2 = self.forward_propagation(X)

        A1 = self.post_processing(A1, self.hidden_h, self.hidden_w, self.dim)
        A2 = self.post_processing(A2, self.new_h, self.new_w, self.dim)

        A2 = self.decompress(A2)
        return A1, A2

    def post_processing(self, X, h, w, d):
        X = X.reshape(h, w, d)
        X = np.round(255 * X)
        X = np.where(X > 255, 255, X)
        X = np.where(X < 0, 0, X)
        return X


if __name__ == '__main__':
    data = load_image('../imgs/dog.png')

    simple_nn = NeuralNetwork(data, c_coeff=3, hidden_layers=(15, 12))
    simple_nn.fit()
    c_output, output = simple_nn.predict(simple_nn.X)

    save_image(output, "RGB", '')



















