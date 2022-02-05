#   ro = distanta de la punct la O(0,0)
#      = sqrt(Xpunct^2 + Ypunct^2)


import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.transform import hough_line
import math


def open_image(path):
    img = mpimg.imread(path)
    # print(img)
    print(img.shape)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Edges')
    plt.imshow(img)

    # return an array of positive points
    arr = []
    for idx_y, y in enumerate(img):
        for idx_x, x in enumerate(y):
            if x == 255:
                arr.append([idx_x, idx_y])
    print(f'positive points shape= {np.array(arr).shape}')
    return arr, img


def calculate_ro(x, y, theta):
    return np.absolute(x * np.cos(theta) + y * np.sin(theta))


def calculate_bias(ro, theta):
    return ro/np.sin(theta)


def hough_line_2(img):
    r_max = math.hypot(img.shape[0], img.shape[1])
    r_dim = 200
    theta_max = np.pi
    theta_dim = 180
    hough_space = np.zeros((r_dim, theta_dim))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 255: continue
            for itheta in range(theta_dim):
                theta = itheta * theta_max / theta_dim
                r = x * math.cos(theta) + y * math.sin(theta)
                ir = r_dim * r / r_max
                hough_space[int(ir), itheta] += 1

    return hough_space


def find_maximum(hough_space, neighborhood_size=20, threshold_proc=0.45):
    threshold = max(map(max, hough_space)) * threshold_proc
    data_max = filters.maximum_filter(hough_space, neighborhood_size)
    maxima = (hough_space == data_max)

    data_min = filters.minimum_filter(hough_space, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

    plt.show()
    plt.imshow(hough_space)
    plt.plot(x, y, 'ro')
    return x, y


def plot_line(img, x, y):
    r_max = math.hypot(img.shape[0], img.shape[1])
    r_dim = 200
    theta_max = np.pi
    theta_dim = 180
    plt.imshow(img)
    for r, theta in zip(y, x):
        # theta = round((j * theta_max) / theta_dim, 1) + 0.01
        theta = (theta * np.pi)/180
        print(f'theta = {theta}\t ro = {r}')
        print(f'b = {r/np.sin(theta)}')
        x_line = np.linspace(0, 100, 2)

        y_line = r/np.sin(theta) - (np.cos(theta)/np.sin(theta)) * x_line
        plt.plot(x_line, y_line)


def hough_transform(arr, img):
    # My Hough
    hough_space = hough_line_2(img)
    # print(f'hough_space.shape= {hough_space.shape}\nro.shape= {ro.shape}\ntheta.shape= {theta.shape}')

    print(f'my hough_space.shape= {hough_space.shape}')
    plt.show()
    plt.imshow(hough_space, origin='lower')
    plt.title('my hough_line')

    plt.show()
    x, y = find_maximum(hough_space)

    plt.show()
    plot_line(img, x, y)

    # Scikit Hough
    tested_angles = np.linspace(-np.pi/2, np.pi/2, 180)
    hough_space, a, b = hough_line(img, tested_angles)
    print(f'hough_space.shape= {hough_space.shape}')
    plt.show()
    plt.title('scikit hough_line')
    plt.imshow(hough_space, origin='lower')

    plt.show()
    x, y = find_maximum(hough_space)

    plt.show()
    plot_line(img, x, y)


if __name__ == '__main__':
    arr, img = open_image('images/edge_simple.bmp')
    hough_transform(arr, img)
    plt.show()