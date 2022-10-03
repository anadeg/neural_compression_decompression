from PIL import Image
import numpy as np


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


if __name__ == '__main__':
    # a = load_image('imgs/sample.bmp')
    # print(a.shape)
    # print(a)
    # a = a.reshape(a.shape[0]*a.shape[1], a.shape[2])
    # print(a.shape)
    # print(a)
    """
    X = [1, 2, 3]
    W = [[0.8, 0.3],
        [-0.2, 0.9]]
    """
    # X = np.array([[1, 1], [2, 2], [3, 3]])
    # W = np.array([[[0.4, 0.5], [-0.1, 0.9]],
    #             [[-0.9, -0.9], [0.3, 0.8]],
    #             [[0.3, -0.3], [0.6, -0.2]]])
    # Y_temp = X.dot(W)

    # X = np.array([1, 2, 2])
    # W = np.array([[0.4, 0.9],
    #               [-0.9, 0.3],
    #               [-0.1, 0.5]])
    # Y_temp = X.dot(W)

    # X = np.array([[1, 1], [2, 2], [3, 3]]).T
    # print(X.shape)
    # W = np.array([[[0.4, 0.4], [0.9, 0.9]],
    #               [[-0.9, -0.9], [0.3, 0.3]],
    #               [[-0.1, -0.1], [0.5, 0.5]]]).T
    # # W = np.array([[0.4, 0.4], [-0.7, -0.7], [-0.2, -0.2]])
    # print(W.shape)
    # print(W, '\n\n')

    # X = np.array([1, 2, 3, 4])
    # W = np.array([[0.2, -0.1],
    #               [0.6, 0.1],
    #               [-0.8, 0.2],
    #               [0.4, -0.4]])

    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).T
    print(X, '\n\n')
    # W = np.array([[[0.2, 0.2], [-0.1, -0.1]],
    #               [[0.6, 0.6], [0.1, 0.1]],
    #               [[-0.8, -0.8], [0.2, 0.2]],
    #               [[0.4, 0.4], [-0.4, -0.4]]]).T
    W = np.array([[0.2, -0.1],
                  [0.6, 0.1],
                  [-0.8, 0.2],
                  [0.4, -0.4]])

    Y_temp = np.dot(X, W)
    print(Y_temp, '\n\n')
    Y_temp = np.matmul(X, W)
    print(Y_temp)
    # Y_temp = np.dot(X, W)

    # print(np.matmul([1, 1], [[0.4, 0.4], [0.9, 0.9]]))



