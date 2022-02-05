# Params:
#       t threshold
#       N number of samples
#           q = probability of a data point to be an inlier => q^s = probability that s points are inliers
#               1 - q^s = probability that in s points at least one is an outlier
#               (1-q^s)^N = probability that in every N selection there is at least a data point which is an outlier
#               (1-q^s)^N = 1 - p <=> N = log(1 - p)/log(1 - q^s)
#           p = probability that all data points in each N samples are inliers
#             = probability with which we want to succeed
#           p ~ 99.9%
#       T = acceptable consensus set
#         = q * n (probability of a point o be an inlier * number of points)
#
#       q ~ 80%

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=2)


def generate_data():
    x = np.arange(-300, 300)
    y = 0.5*x + 50
    data = np.column_stack([x, y])

    data_faulty = np.array(40 * [(200.0, -130)])
    data_faulty += 4 * np.random.normal(size=data_faulty.shape)
    data_faulty = data[:data_faulty.shape[0]]

    data_noise = np.random.normal(size=data.shape)
    data += 2 * data_noise
    data[::2] += 3*data_noise[::2]
    data[::5] += 100*data_noise[::5]
    plt.plot(data[:, 0], data[:, 1], '.')
    plt.show()
    return data


def calculate_N(q, s, p=0.999):
    return np.ceil(np.log10((1 - p)) / np.log10(1 - np.power(q, s))).astype(np.int64)


def calculate_T(q, n):
    return np.ceil(q * n).astype(np.int64)


def calculate_distance(data_point, a, b, c):
    return (np.absolute(a*data_point[0] + b*data_point[1] + c))/np.sqrt(np.square(a) + np.square(b))


def plot_data(data):
    plt.plot(data[:, 0], data[:, 1], '.')
    return


def plot_line(data, a, b, c):
    # x = np.linspace(min(data[:, 0], max(data[:, 0])))
    x = np.linspace(-300, 300)
    y = (-a*x - c) / b
    plt.plot(x, y, '-r')
    return


def ransac(data, d=10, q=0.8, p=0.999, s=2):
    d += 2
    N = calculate_N(q, s, p)
    T = calculate_T(q, data.shape[0])
    random_points_indexes = np.random.choice(np.arange(0, data.shape[0]), replace=False, size=(N, 2))
    max_number_inliers = 0
    a_fin = 1
    b_fin = 1
    c_fin = 1
    for i in random_points_indexes:
        number_inliers = 0
        a = data[i[0]][1] - data[i[1]][1]       # y1 - y2
        b = data[i[1]][0] - data[i[0]][0]       # x2 - x1
        c = data[i[0]][0] * data[i[1]][1] - data[i[1]][0] * data[i[0]][1]   # x1*y2 - x2*y1
        for data_point in data:
            dist = calculate_distance(data_point, a, b, c)
            if dist <= d:
                number_inliers += 1
        if number_inliers > max_number_inliers:
            max_number_inliers = number_inliers
            a_fin = a
            b_fin = b
            c_fin = c
    plot_data(data)
    plot_line(data, a_fin, b_fin, c_fin)
    plt.show()
    return


if __name__ == '__main__':
    data = generate_data()
    print(data.shape)
    ransac(data)
