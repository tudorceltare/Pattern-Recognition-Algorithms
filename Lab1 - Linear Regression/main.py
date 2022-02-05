import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def read_data(src):
    data = np.loadtxt(src, skiprows=0)
    return data


def linear_regression_2(data: np.ndarray, src, epochs=1, alpha=0.01):
    scaler = StandardScaler()
    scaler.fit(data)
    data1 = scaler.transform(data)
    # data1 = data
    x = data1[:, 0]
    x = np.vstack((np.ones(len(x)), x)).T
    y = data1[:, 1]
    y = y.reshape(len(y), 1)

    figure, axis = plt.subplots(2)
    figure.suptitle(f'Linear regression and loss for {src}')
    loss_list = []
    theta = np.zeros((2, 1))
    n = len(data)
    # n = 3
    for i in range(epochs):
        y_pred = np.dot(x, theta)

        # we calculate the loss only to append it to the loss_list and display it in a graph
        loss = (1 / (2 * n)) * np.sum(np.square(y_pred - y))
        print(f'epoch {i+1}\ttheta0 = {theta[0]}, \ttheta1 = {theta[1]}, \tloss = {loss}')

        loss_list.append(loss)
        d_theta = (1 / (2*n)) * np.dot(x.transpose(), y_pred - y)
        theta = theta - alpha * d_theta

    plotX = np.linspace(min(x[:, 1]), max(x[:, 1]))
    # plotX = np.linspace(min(data[:, 0]), max(data[:, 0]))
    plotY = theta[0] + theta[1]*plotX
    rng = np.arange(0, epochs)
    axis[0].scatter(x[:, 1], y)
    axis[0].set_title('Regression')
    axis[0].set_xlabel('X')
    axis[0].set_ylabel('Y')
    axis[0].plot(plotX, plotY, '-r')
    axis[1].grid()
    axis[1].set_title('Loss')
    axis[1].plot(rng, loss_list)
    axis[1].set_xlabel('epochs')
    axis[1].set_ylabel('loss')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


if __name__ == "__main__":
    src = 'points5.txt'
    linear_regression_2(read_data('data/' + src), src, epochs=300, alpha=0.066)
