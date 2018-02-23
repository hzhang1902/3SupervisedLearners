import numpy as np

# Please include images.py and labels.py
# Usage:
# images: all images in vector form
# labels: all labels in on-hot encoding
# matrix_image: a confusion matrix of 10 classes with all images
# matrix_label: a confusion matrix of 10 classes with all labels
# matrix_image_test, matrix_image_validation, matrix_image_training: split into sets
# matrix_label_test, matrix_label_validation, matrix_label_training: split into sets


IMAGE_SIZE = 28*28
IMAGE_NUM = 6500
LABEL_MAX = 9

# read files
raw_labels = np.load('labels.npy')
raw_images = np.load('images.npy')

# convert image to vector
images = []
for raw_image in raw_images:
    image = np.reshape(raw_image, IMAGE_SIZE)
    images.append(image)

# convert label to 1-hot encoding
labels = np.zeros((IMAGE_NUM, LABEL_MAX + 1))
labels[np.arange(IMAGE_NUM), raw_labels] = 1

# split into 10 classes
index_image = 0
class_image = []
class_label = []
matrix_image = []
matrix_label = []
while index_image < IMAGE_NUM:
    class_image.append(images[index_image])
    class_label.append(labels[index_image])
    new_class = (index_image + 1) % (IMAGE_NUM / 10) == 0
    if new_class:
        matrix_image.append(class_image)
        matrix_label.append(class_label)
        class_image = []
        class_label = []
    index_image += 1

# split into sets
matrix_image_test = []
matrix_image_validation = []
matrix_image_training = []
matrix_label_test = []
matrix_label_validation = []
matrix_label_training = []

index_class = 0
while index_class < 10:
    index_image = 0
    matrix_image_training_class = []
    matrix_image_validation_class = []
    matrix_image_test_class = []

    matrix_label_training_class = []
    matrix_label_validation_class = []
    matrix_label_test_class = []

    while index_image < IMAGE_NUM / 10:
        image = matrix_image[index_class][index_image]
        label = matrix_label[index_class][index_image]
        if index_image < IMAGE_NUM / 10 * 0.6:
            matrix_image_training_class.append(image)
            matrix_label_training_class.append(label)
        elif index_image < IMAGE_NUM / 10 * 0.75:
            matrix_image_validation_class.append(image)
            matrix_label_validation_class.append(label)
        else:
            matrix_image_test_class.append(image)
            matrix_label_test_class.append(label)

        index_image += 1

    matrix_image_training.append(matrix_image_training_class)
    matrix_image_validation.append(matrix_image_validation_class)
    matrix_image_test.append(matrix_image_test_class)

    matrix_label_training.append(matrix_label_training_class)
    matrix_label_validation.append(matrix_label_validation_class)
    matrix_label_test.append(matrix_label_test_class)

    index_class += 1


# tests
"""
print(matrix_label_training.__len__(),
      matrix_label_training[0].__len__(),
      matrix_label_training[0][0],
      matrix_label_training[0][0].__len__())

print(matrix_image_test.__len__(),
      matrix_image_test[0].__len__(),
      matrix_image_test[0][0],
      matrix_image_test[0][0].__len__())
"""
