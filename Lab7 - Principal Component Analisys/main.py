import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from tqdm import tqdm


def read_data(path):
    return np.loadtxt(path, skiprows=0)


def get_mean_vector(data):
    mean = []
    for i in range(data.shape[1]):
        mean.append(np.mean(data[:, i]))
    return np.array(mean)


def covariance_matrix(new_data):
    cov = new_data.transpose().dot(new_data)/(new_data.shape[0] - 1)
    plt.title('Covariance Matrix')
    plt.imshow(cov)
    plt.show()
    return cov


def normalized_data(data):
    mean = get_mean_vector(data)
    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        new_data[:, i] = data[:, i] - mean[i]
    return new_data


def eigenvalues(matrix):
    ev, q = np.linalg.eig(matrix)
    print(f"eigenvalue: {ev[0]}\t Q: {q.transpose()[0,1]}")
    print(q.transpose())
    return q.transpose()


def pca_coeff(data):
    return data.dot(eigenvalues(covariance_matrix(data)))


def pca_matrix(data, eigen_matrix, k=1):
    print(data[0, 0])
    sub_mat = eigen_matrix[:, :k]
    xk = data.dot(sub_mat.dot(sub_mat.transpose()))
    print(xk.shape)
    return xk


if __name__ == '__main__':
    data = read_data('data/pca3d.txt')
    data = normalized_data(data)

    xk = pca_matrix(data, eigenvalues(covariance_matrix(data)))
    print(xk[0, 0])
    # print(data[0, 0] - xk[0, 0])

    print(f"avg: {np.mean(data - xk)}")
    #
    # coeff = pca_coeff(data)
    # plt.scatter(xk[:, 0], coeff[:, 0])
    # plt.show()

    # print(get_mean_vector(data))
