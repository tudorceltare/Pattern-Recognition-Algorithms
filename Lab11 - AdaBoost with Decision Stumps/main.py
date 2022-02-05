import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from typing import Optional
from sklearn.tree import DecisionTreeClassifier
import os
import numpy as np
import random
from tqdm import tqdm


def open_image(path):
    return mpimg.imread(path)


def print_points(image):
    """
    image[:,:,0] => blue
    image[:,:,2] => red
    """
    for i in range(image.shape[2]):
        if i % 2 == 0:
            coord = np.argwhere(image[:, :, i] == 0)
            plt.scatter(coord[:, 1], coord[:, 0])
    plt.imshow(image)
    plt.show()


def get_training_set(image):
    coord_blue = np.argwhere(image[:, :, 0] == 0)
    coord_red = np.argwhere(image[:, :, 2] == 0)
    X = np.concatenate((coord_red, coord_blue), axis=0)
    y = np.concatenate((np.ones(len(coord_red)), np.ones(len(coord_blue)) * -1), axis=None)
    return X, y


def print_prediction(X, y, clf, image, T):
    y_predicted = clf.predict(X)
    correct_point_red = []
    correct_point_blue = []
    incorrect_point_red = []
    incorrect_point_blue = []
    for point, prediction, truth in zip(X, y_predicted, y):
        if prediction == truth and truth == 1:
            correct_point_red.append(point)
        elif prediction == truth and truth == -1:
            correct_point_blue.append(point)
        elif prediction != truth and truth == 1:
            incorrect_point_red.append(point)
        elif prediction != truth and truth == -1:
            incorrect_point_blue.append(point)
    correct_point_red = np.array(correct_point_red)
    correct_point_blue = np.array(correct_point_blue)
    incorrect_point_red = np.array(incorrect_point_red)
    incorrect_point_blue = np.array(incorrect_point_blue)
    plt.title(f'T = {T}')
    plt.scatter(correct_point_red[:, 1], correct_point_red[:, 0], marker='.', c='red')
    plt.scatter(correct_point_blue[:, 1], correct_point_blue[:, 0], marker='.', c='blue')
    if incorrect_point_red.size != 0:
        plt.scatter(incorrect_point_red[:, 1], incorrect_point_red[:, 0], marker='*', c='green')
    if incorrect_point_blue.size != 0:
        plt.scatter(incorrect_point_blue[:, 1], incorrect_point_blue[:, 0], marker='*', c='yellow')
    plt.imshow(image)
    plt.show()


class AdaBoost:
    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.w = None

    def fit(self, X_train, y_train, T=10):
        n = X_train.shape[0]
        self.w = np.zeros(shape=(T, n))
        self.stumps = np.zeros(shape=T, dtype=object)
        self.stump_weights = np.zeros(shape=T)
        self.errors = np.zeros(shape=T)

        self.w[0] = np.ones(X_train.shape[0]) / X_train.shape[0]
        for t in tqdm(range(T)):
            # fit stumps and predict with stumps
            current_w = self.w[t]
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(X_train, y_train, sample_weight=current_w)
            stump_predict = stump.predict(X_train)

            # error = sum of all weights where stump prediction is inaccurate
            err = current_w[(stump_predict != y)].sum()
            stump_weight = np.log((1 - err) / err) / 2

            # update weights
            new_w = (current_w * np.exp(-stump_weight * y_train * stump_predict))
            new_w /= new_w.sum()

            # if not final stump, update weights
            if t + 1 < T:
                self.w[t + 1] = new_w

            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err

        return self

    def predict(self, X):
        # H(x) = sign(sum from t=1 to T(stump_weight * stump_pred))
        stump_predictions = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_predictions))


if __name__ == '__main__':
    image = open_image('images/points4.bmp')
    T = 3
    X, y = get_training_set(image)
    clf = AdaBoost().fit(X_train=X, y_train=y, T=T)
    print_prediction(X, y, clf, image, T)

    train_err = (clf.predict(X) != y).mean() * 100
    print(f'Train error: {train_err:.2f}%')


