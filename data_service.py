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
labels = raw_labels
"""
labels = np.zeros((IMAGE_NUM, LABEL_MAX + 1))
labels[np.arange(IMAGE_NUM), raw_labels] = 1
"""

# split into 10 classes
index_image = 0
index_class = 0
matrix_image = []
matrix_label = []
while index_class < 10:
    class_image = []
    class_label = []
    matrix_image.append(class_image)
    matrix_label.append(class_label)
    index_class += 1

while index_image < IMAGE_NUM:
    label = labels[index_image]
    matrix_image[label].append(images[index_image])
    matrix_label[label].append(labels[index_image])
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

    class_size = matrix_image[index_class].__len__()
    while index_image < class_size:
        image = matrix_image[index_class][index_image]
        label = matrix_label[index_class][index_image]
        if index_image < class_size * 0.6:
            matrix_image_training.append(image)
            matrix_label_training.append(label)
        elif index_image < class_size * 0.75:
            matrix_image_validation.append(image)
            matrix_label_validation.append(label)
        else:
            matrix_image_test.append(image)
            matrix_label_test.append(label)

        index_image += 1
    index_class += 1


# tests
"""
print(matrix_label_training.__len__(), matrix_label_validation.__len__())

print(matrix_image_test.__len__())
"""