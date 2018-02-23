import numpy as np

IMAGE_SIZE = 28*28
IMAGE_NUM = 6500
LABEL_MAX = 10

raw_labels = np.load('labels.npy')
raw_images = np.load('images.npy')

images = []
for raw_image in raw_images:
    image = np.reshape(raw_image, IMAGE_SIZE)
    images.append(image)

labels = np.zeros((IMAGE_NUM, LABEL_MAX + 1))
labels[np.arange(IMAGE_NUM), raw_labels] = 1

# images
# labels
