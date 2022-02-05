import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import random
from tqdm import tqdm


def open_image(path):
    return mpimg.imread(path)


def get_histogram(image, m=8, show=False):
    colors = ['Red', 'Green', 'Blue']
    hist = []
    for channel_id, c in zip(range(image.shape[2]), colors):
        histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=m, range=(0, 256))
        if show:
            plt.title(c)
            plt.bar(bin_edges[:-1], histogram, width=20, color=c)
            plt.show()
        hist.append(histogram)
    return np.array(hist)


def get_training_hist(path, m=8):
    folders = os.listdir(path)
    training_set = []
    labels = []
    y = 0
    print('Reading training set...')
    for folder in tqdm(folders):
        images = os.listdir(path + '/' + folder)
        y += 1
        for image_path in images:
            # print(path + '/' + folder + '/' + image_path)
            labels.append(y)
            training_set.append(get_histogram(open_image(path + '/' + folder + '/' + image_path), m))
    return np.array(training_set), np.array(labels)


def calculate_distance_vector(test_image, training_set, labels):
    test_hist = get_histogram(test_image)
    distance_vector = []
    for hist, label in zip(training_set, labels):
        dist = 0
        for c in range(hist.shape[0]):
            dist += np.linalg.norm(hist[c] - test_hist[c])
        distance_vector.append([dist, label])
    # print(np.array(distance_vector).shape)
    distance_vector = np.array(distance_vector)
    p = distance_vector[np.argsort(distance_vector[:, 0])]
    return p


def k_nearest_neighbours(training_path, training_set, training_labels, test_image, k=5, show=False):
    p = calculate_distance_vector(test_image, training_set, training_labels)
    files = os.listdir(training_path)
    final_hist = np.zeros(len(files))
    for i in range(k):
        final_hist[int(p[i][1]) - 1] += 1
    if show:
        plt.title(f'Prediction: {files[np.argmax(final_hist)]}')
        plt.imshow(test_image)
        plt.show()
        print(f'Predicted class: {files[np.argmax(final_hist)]}')
        normalized = (final_hist - min(final_hist))/(max(final_hist) - min(final_hist))
        print(f'Prediction: {normalized}')
    # returns the predicted class of the image as an index
    return np.argmax(final_hist)


def test_knn(train_path, test_path, k=5, show=False):
    train_set, train_labels = get_training_hist(train_path)
    test_folders = os.listdir(test_path)
    test_images = []
    test_labels = []
    y = 0
    for folder in test_folders:
        images = os.listdir(test_path + '/' + folder)
        y += 1
        for image_path in images:
            test_labels.append(y)
            test_images.append(open_image(test_path + '/' + folder + '/' + image_path))
    test_labels = np.array(test_labels)
    test_images = np.array(test_images)
    confusion_matrix = np.zeros((len(test_folders), len(test_folders)))
    for test_image, label in tqdm(zip(test_images, test_labels)):
        prediction = k_nearest_neighbours(train_path, train_set, train_labels, test_image, k)
        confusion_matrix[label - 1][int(prediction)] += 1

    # print accuracy
    acc = np.trace(confusion_matrix)/confusion_matrix.sum()
    print(f'Accuracy: {acc}')
    if show:
        plt.title(f'Confusion matrix for K = {k}')
        plt.imshow(confusion_matrix)
        plt.colorbar()
        plt.show()
    return acc


if __name__ == '__main__':
    # example 1
    # image = open_image('./train/city/000005.jpeg')
    # get_histogram(image, show=True)

    # example 2
    # training_set, training_labels = get_training_hist(path='./train')
    # image = open_image('./test/city/000003.jpeg')
    # k_nearest_neighbours(training_path='./train',
    #                      training_set=training_set,
    #                      training_labels=training_labels,
    #                      test_image=image,
    #                      k=15, show=True)

    # example 3
    test_knn('./train', './test', k=5, show=True)

