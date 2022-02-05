import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from tqdm import tqdm


def open_image(path):
    return mpimg.imread(path)


def load_images(path):
    # loads images from folder at location 'path'
    images = []
    for filename in os.listdir(path):
        images.append(mpimg.imread(os.path.join(path, filename)))
    return np.array(images)


def expectation(images):
    # this function returns a 2d matrix of mean values for each pixel in all images
    # the returned matrix shape is image.shape[0] X image.shape[1]
    mean_val = np.zeros(images[0].shape)
    for img in images:
        mean_val += img
    mean_val /= len(images)
    plt.title('Mean Matrix')
    plt.imshow(mean_val)
    plt.show()
    return mean_val


def standard_deviation(images, mean_val):
    std = np.zeros(images[0].shape)
    for img in images:
        std += (img - mean_val) ** 2
    std /= len(images)
    std = np.sqrt(std)
    plt.title('Standard Deviation')
    plt.imshow(std)
    plt.show()
    return std


def covariance_matrix(images, mean_val):
    cm = np.zeros((mean_val.shape[0] ** 2, mean_val.shape[1] ** 2))
    mean_vec = mean_val.flatten()
    for img in tqdm(images):
        img_vec = img.flatten()
        len_r = len(img_vec)
        img_vec = img_vec - mean_vec
        for i in range(len_r):
            cm[i] += img_vec[i] * img_vec
    cm /= len(images)
    plt.title('Covariance Matrix')
    plt.imshow(cm)
    plt.show()
    print(cm)
    return cm


def correlation_coeff(cm, std, yi, xi, yj, xj):
    i = yi * std.shape[1] + xi
    j = yj * std.shape[1] + xj
    c_coeff = cm[i][j] / (std[yi][xi] * std[yj][xj])
    print(f"c_coeff({yi}, {xi})({yj}, {xj}) = {c_coeff}")
    return c_coeff


def correlation_graph(images, yi, xi, yj, xj):
    i = yi * std.shape[1] + xi
    j = yj * std.shape[1] + xj
    corr_graph = np.ones((256, 256)) * 255
    for img in images:
        img_vec = img.flatten()
        corr_graph[img_vec[i], img_vec[j]] = 0
    plt.title('Correlation Graph')
    plt.imshow(corr_graph)
    plt.show()


def probability_density_graph(mean_val, std, y_pos, x_pos):
    x = np.linspace(0, 255, 256, dtype=int)
    e = -((x - mean_val[y_pos][x_pos]) ** 2) / (2 * (std[y_pos][x_pos] ** 2))
    y = (1 / (np.sqrt(2 * np.pi) * std[y_pos][x_pos])) \
           * np.exp(e)
    print(y)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # image = open_image('./images/face00001.bmp')
    images = load_images('./images')
    mean_val = expectation(images)
    std = standard_deviation(images, mean_val)
    cm = covariance_matrix(images, mean_val)
    # correlation_coeff(cm, std, 10, 3, 9, 15)
    # correlation_coeff(cm, std, 5, 4, 18, 0)
    # correlation_graph(images, 10, 3, 9, 15)
    # correlation_graph(images, 5, 4, 18, 0)
    probability_density_graph(mean_val, std, 0, 0)
