import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
import random


def open_image(path):
    return mpimg.imread(path)


def print_points(image):
    coord_blue = np.argwhere(image[:, :, 0] == 0)
    coord_red = np.argwhere(image[:, :, 2] == 0)
    plt.imshow(image)
    plt.scatter(coord_blue[:, 1], coord_blue[:, 0])
    plt.scatter(coord_red[:, 1], coord_red[:, 0])
    plt.show()


def get_training_set(image):
    coord_blue = np.argwhere(image[:, :, 0] == 0)
    coord_red = np.argwhere(image[:, :, 2] == 0)
    X = np.concatenate((coord_red, coord_blue), axis=0)
    y = np.concatenate((np.zeros(len(coord_red)), np.ones(len(coord_blue))), axis=None)
    return X, y


def step_function(z):
    return 1.0 if (z > 0) else 0.0


def step_function_batch(z):
    return np.where(z > 0.0, 1.0, 0.0)


def perceptron(X, y, lr, epochs):
    m, n = X.shape
    theta = np.random.uniform(low=0.001, high=1, size=(n + 1, 1))

    # the list of missed predictions
    n_missed_list = []
    for _ in tqdm(range(epochs)):
        n_miss = 0
        for idx, x_i in enumerate(X):
            # we insert 1  for x^0 which is x0
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

            # Predict by calculating hypothesis
            y_hat = step_function(np.dot(theta.T, x_i))

            if (np.squeeze(y_hat) - y[idx]) != 0:
                # Updating theta values when prediction is wrong
                theta += lr * ((y[idx] - y_hat) * x_i)
                n_miss += 1
        n_missed_list.append(n_miss)
    return theta, n_missed_list


def calculate_gradients(X, y, yhat):
    return (1 / X.shape[0]) * np.dot(X.T, (yhat - y))


def perceptron_batch(X, y, lr, epochs):
    m, n = X.shape
    # theta.shape = (m, n+1, 1). Ex theta[i] = [[0.19570757], [0.06853483], [0.96043971]]
    theta = np.random.uniform(low=0.001, high=1, size=(m, n + 1))

    # appended 1 to the end of each row of the X dataset
    X_ = np.c_[X, np.ones(m)]

    # Loss
    loss_list = []
    for epoch in tqdm(range(epochs)):
        yhat = X_ * theta
        yhat = yhat.sum(axis=1)
        yhat = step_function_batch(yhat)
        theta -= lr * calculate_gradients(X_, y, yhat)
    return theta


############################ new try ###########################
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# def loss(y, y_hat):
#     loss = -np.mean(y * (np.log(y_hat)) - (1 - y) * np.log(1 - y_hat))
#     return loss

def loss(y, y_hat):
    loss = -np.mean(y**2 - y_hat**2)
    return loss


def gradients(X, y, y_hat):
    # m-> number of training examples.
    m = X.shape[0]

    # Gradient of loss w.r.t weights.
    dw = (1 / X.shape[0]) * np.dot(X.T, (y_hat - y))

    # Gradient of loss w.r.t bias.
    db = (1 / X.shape[0]) * np.sum((y_hat - y))

    return dw, db


def normalize(X):
    m, n = X.shape

    # Normalizing all the n features of X.
    for i in range(n):
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X


def perceptron_batch_1(X, y, bs, lr, epochs):
    m, n = X.shape

    # Initializing weights and bias to zeros.
    theta = np.random.uniform(low=0.001, high=1, size=(n + 1, 1))
    # theta = np.zeros((n+1, 1))

    # Reshaping y.
    y = y.reshape(m, 1)

    # Normalizing the inputs.
    # x = normalize(X)
    x = X

    # Empty list to store losses.
    losses = []

    # Training loop.
    for epoch in range(epochs):
        for i in range((m - 1) // bs + 1):
            # Defining batches. SGD.
            start_i = i * bs
            end_i = start_i + bs
            xb = x[start_i:end_i]
            yb = y[start_i:end_i]

            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(xb, theta[:-1]) + theta[-1])

            # Getting the gradients of loss w.r.t parameters.
            dw, db = gradients(xb, yb, y_hat)

            # Updating the parameters.
            theta[:-1] -= lr * dw
            theta[-1] -= lr * db

        # Calculating loss and appending it in the list.
        l = loss(y, sigmoid(np.dot(X, theta[:-1]) + theta[-1]))
        losses.append(l)

    # returning weights, bias and losses(List).
    return theta, losses


def plot_decision_boundary(X, theta, loss_val, image):
    x1 = [min(X[:, 0]), max(X[:, 0])]
    m = -theta[1] / theta[2]
    c = -theta[0] / theta[2]
    x2 = m * x1 + c

    # Plotting
    plt.imshow(image)
    plt.plot(X[:, 1][y == 0], X[:, 0][y == 0], "r^")
    plt.plot(X[:, 1][y == 1], X[:, 0][y == 1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Perceptron Algorithm')
    plt.plot(x2, x1, 'g-')
    plt.show()
    if loss_val:
        plt.title('Loss values per epoch')
        plt.xlabel('Epochs')
        plt.plot(loss_val)
        plt.show()


def plot_decision_boundary_2(X, theta, loss_val, image):
    print(theta)
    w = theta[:-1]
    b = theta[-1]
    x1 = [min(X[:, 0]), max(X[:, 0])]
    m = -w[0] / w[1]
    c = -b / w[1]
    x2 = m * x1 + c

    # Plotting
    plt.imshow(image)
    plt.plot(X[:, 1][y == 0], X[:, 0][y == 0], "g^")
    plt.plot(X[:, 1][y == 1], X[:, 0][y == 1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')
    plt.plot(x2, x1, 'y-')
    plt.show()
    if loss_val:
        plt.title('Loss values per epoch')
        plt.xlabel('Epochs')
        plt.plot(loss_val)
        plt.show()


if __name__ == '__main__':
    image = open_image('./test/test06.bmp')
    X, y = get_training_set(image)
    # theta, miss_l = perceptron(X, y, 0.001, 10)
    theta, losses = perceptron_batch_1(X, y, X.shape[0], 0.001, 330)
    # plot_decision_boundary(X, theta, losses, image)
    plot_decision_boundary_2(X, theta, losses, image)
    # print(f'final missed: {miss_l[-1]}')
