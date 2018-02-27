import knn_data_service as ds
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt


def get_miss_classified(images, labels, knn):
    miss_classified = []
    preds = knn.predict(images)
    index_c = 0
    while index_c < labels.__len__():
        if not preds[index_c] == labels[index_c]:
            miss_classified.append(index_c)
        index_c += 1
    return miss_classified


def get_accuracy(confusion_matrix, total_num):
    accurate = 0
    index_a = 0
    while index_a < 10:
        accurate += confusion_matrix[index_a][index_a]
        index_a += 1
    return accurate/total_num


def get_best_knn(t_images, t_labels, v_images, v_labels, max_num):
    best = 0
    b_knn = None
    index_k = 1
    while index_k <= max_num:
        knn = KNeighborsClassifier(index_k)
        knn.fit(t_images, t_labels)
        score = knn.score(v_images, v_labels)
        print(index_k, score)
        if score > best:
            best = score
            b_knn = knn
        index_k += 1
    return b_knn


def get_confusion_matrix(images, labels, knn):
    confusion_matrix = []
    x_index = 0
    while x_index < 10:
        confusion_row = []
        y_index = 0
        while y_index < 10:
            confusion_row.append(0)
            y_index += 1
        confusion_matrix.append(confusion_row)
        x_index += 1

    x_index = 0
    preds = knn.predict(images)
    while x_index < labels.__len__():
        actual = labels[index]
        prediction = preds[index]
        confusion_matrix[actual][prediction] += 1
        x_index += 1

    return confusion_matrix


def show_image(image):
    index_i = 0
    img = []
    buffer = []
    while index_i < image.__len__():
        buffer.append(image[index_i])
        if buffer.__len__() == 28:
            img.append(buffer)
            buffer = []
        index_i += 1
    plt.imshow(img)
    plt.colorbar()
    plt.show()


training_images = ds.matrix_image_training
training_labels = ds.matrix_label_training
validation_images = ds.matrix_image_validation
validation_labels = ds.matrix_label_validation
test_images = ds.matrix_image_test
test_labels = ds.matrix_label_test

# best neighbor num
best_knn = get_best_knn(training_images, training_labels, validation_images, validation_labels, 20)
# best_knn = KNeighborsClassifier(1)
# best_knn.fit(training_images, training_labels)

# confusion matrix
confusion = get_confusion_matrix(test_images, test_labels, best_knn)
print(confusion)

# test set accuracy
accuracy = get_accuracy(confusion, test_labels.__len__())
print(accuracy)

# missclassified
mc = get_miss_classified(test_images, test_labels, best_knn)

# number of missclassified
print(mc.__len__())

# 3 missclassified examples
mc1 = mc[2]
mc2 = mc[64]
mc3 = mc[39]

# prob and interpreted result
print(best_knn.predict_proba([test_images[mc1]]), test_labels[mc1])
print(best_knn.predict_proba([test_images[mc2]]), test_labels[mc2])
print(best_knn.predict_proba([test_images[mc3]]), test_labels[mc3])

# visualization
show_image(test_images[mc1])
show_image(test_images[mc2])
show_image(test_images[mc3])

# nearest neighbors
index = 0
distances, indices = best_knn.kneighbors([test_images[mc1]], 3, True)
while index < 3:
    distance = distances[0][index]
    indice = indices[0][index]
    print(training_labels[indice], distance)
    index += 1

index = 0
distances, indices = best_knn.kneighbors([test_images[mc2]], 3, True)
while index < 3:
    distance = distances[0][index]
    indice = indices[0][index]
    print(training_labels[indice], distance)
    index += 1

index = 0
distances, indices = best_knn.kneighbors([test_images[mc3]], 3, True)
while index < 3:
    distance = distances[0][index]
    indice = indices[0][index]
    print(training_labels[indice], distance)
    index += 1


