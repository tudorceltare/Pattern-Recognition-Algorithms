import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np


def open_image(path):
    return mpimg.imread(path)


def dt(image):
    transform_image = image
    temp_down = {
        # (y, x)
        # first line
        (-1, 0): 1,
        (-1, -1): 1.41,
        (-1, 1): 1.41,
        # second line
        (0, -1): 1
    }

    down_weight = {
        # (y, x)
        # first line
        (-1, 0): 2,
        (-1, -1): 3,
        (-1, 1): 3,
        # second line
        (0, -1): 2
    }

    temp_up = {
        # (y, x)
        (0, 1): 1,
        # third line
        (1, -1): 1.41,
        (1, 0): 1,
        (1, 1): 1.41
    }

    up_weight = {
        # (y, x)
        (0, 1): 2,
        # third line
        (1, -1): 3,
        (1, 0): 2,
        (1, 1): 3
    }

    for j in range(1, image.shape[0]):
        for i in range(1, image.shape[1] - 1):
            minimum = transform_image[j][i]
            for (k, l) in down_weight:
                temp_down[(k, l)] = transform_image[j + k][i + l] + down_weight[(k, l)]
            key_min = min(temp_down.keys(), key=(lambda t: temp_down[t]))
            if temp_down[key_min] < minimum:
                minimum = temp_down[key_min]
            transform_image[j][i] = minimum

    for j in range(image.shape[0] - 2, 0, -1):
        for i in range(image.shape[1] - 2, 0, -1):
            minimum = transform_image[j][i]
            for (k, l) in up_weight:
                temp_up[(k, l)] = transform_image[j + k][i + l] + up_weight[(k, l)]
            key_min = min(temp_up.keys(), key=(lambda t: temp_up[t]))
            if temp_up[key_min] < minimum:
                minimum = temp_up[key_min]
            transform_image[j][i] = minimum

    return transform_image


def get_center_of_mass(image):
    sum_x = 0
    sum_y = 0
    N = 0
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if image[j][i] == 0:
                sum_x += i
                sum_y += j
                N += 1
    return round(sum_x / N), round(sum_y / N)


def model_matching(image1, image2):
    score = 0.0
    cm1 = get_center_of_mass(image1)
    cm2 = get_center_of_mass(image2)
    delta_y = abs(cm1[0] - cm2[0])
    delta_x = abs(cm1[1] - cm2[1])

    transform = dt(image1)
    T = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    img_translation = cv2.warpAffine(image2, T, (image2.shape[1], image2.shape[0]), borderValue=255)

    contour_length = 0
    for j in range(img_translation.shape[0]):
        for i in range(img_translation.shape[1]):
            if img_translation[j][i] == 0:
                contour_length += 1
                score += transform[j][i]

    # percentage = (value - 0)/(255 * contour_length - 0)
    # (score/(255 * contour_length)) * 100.0
    percentage = (1 - (score/(255 * contour_length))) * 100.0
    print(f"score: \t\t{score/contour_length}")
    print("matching: \t"+"{:.2f}".format(percentage)+"%")
    return score


def show_image(image, text="image"):
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(text)
    plt.imshow(image)
    plt.show()


def trying(image):
    maxim = 0
    for j in range(image.shape[0]):
        if maxim < max(image[j]):
            maxim = max(image[j])
    print(maxim)


if __name__ == '__main__':
    image = open_image('images/PatternMatching/template.bmp')
    image2 = open_image('images/PatternMatching/unknown_object1.bmp')
    print(image.shape)
    show_image(image, "Edges")
    show_image(image2, "Edges")

    image = dt(image)
    show_image(image, 'Distance transform')
    model_matching(image, image2)
    trying(image)
