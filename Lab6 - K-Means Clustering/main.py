import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random


def sqrt_sum_T(data):
    a, b = data[1]
    return np.sqrt(np.sum((a - b) ** 2, axis=0))


def open_image(path):
    return mpimg.imread(path)


def print_points(image):
    """
    Prints points for an image that has data only in 1 dimension
    """
    coord = np.argwhere(image[:, :, 0] == 0)
    plt.imshow(image)
    plt.scatter(coord[:, 1], coord[:, 0])
    plt.show()


def select_k_random_points(image, k=1):
    """
    Returns a list of k randomly selected points from a coordinates list
    """
    coord_list = np.argwhere(image[:, :, 0] == 0)
    indexes = random.sample(range(len(coord_list)), k)
    k_rand = []
    for i in indexes:
        k_rand.append(coord_list[i])
    return np.array(k_rand)


def select_k_random_points_balanced(image, k=1):
    """
    Returns a list of k randomly selected points that are equally spaced apart from a coordinates list
    """
    coord_list = np.argwhere(image[:, :, 0] == 0)
    splited = np.array_split(coord_list, k)
    k_rand = []
    for array in splited:
        k_rand.append(random.choice(array))
    return np.array(k_rand)


def assignment(coord, centers):
    changes = 0
    for point, l in zip(coord[:, :-1], range(len(coord))):
        dist = []
        for center, i in zip(centers, range(len(centers))):
            dist.append([np.linalg.norm(point - center), i])
        dist = np.array(dist)
        min_dist = dist[np.where(dist[:, 0] == min(dist[:, 0]))]
        if coord[l][-1] != min_dist[0][1]:
            coord[l][-1] = min_dist[0][1]
            changes += 1

    return coord, changes


def update(centers, coord_list):
    sum = np.zeros((len(centers), coord_list.shape[1] - 1))
    k_count = np.zeros(len(centers))
    for point in coord_list:
        k_index = int(point[2])
        sum[k_index] += point[:2]
        k_count[k_index] += 1
    for i in range(len(centers)):
        sum[i] /= k_count[i]
    return sum


def k_means(image, k=1, balanced=False):
    coord_list = np.argwhere(image[:, :, 0] == 0)
    # adds a third column which contains the class of the point
    # by default it is class 0
    coord_list = np.append(coord_list, np.zeros(len(coord_list)).reshape(len(coord_list), 1), axis=1)
    if balanced:
        centers = select_k_random_points_balanced(image, k)
    else:
        centers = select_k_random_points(image, k)

    # while no change
    changes = 1
    while changes > 0:
        print(f"Changes: {changes}")
        coord_list, changes = assignment(coord_list, centers)
        centers = update(centers, coord_list)

    draw_classes(image, coord_list, k, balanced)
    return centers, coord_list


def draw_classes(image, coord, k, balanced):
    for center in range(k):
        indexes = np.where(coord[:, 2] == center)[0]
        data_array = []
        for i in indexes:
            data_array.append(coord[i])
        data_array = np.array(data_array)
        plt.scatter(data_array[:, 1], data_array[:, 0])
    if balanced:
        plt.title(f'K = {k} with Balanced init')
    else:
        plt.title(f"K = {k} with Random init")
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    image = open_image('./images/points2.bmp')
    centers, coord = k_means(image, 4, balanced=True)
