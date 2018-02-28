import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.cross_validation import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

image_data = np.load('images.npy')
label_data = np.load('labels.npy')


data = image_data.reshape([6500,784])
labels = keras.utils.to_categorical(label_data, num_classes=10)

#Splitting Data into training, valdation, and test sets
data_train, data_rest, labels_train, labels_rest = train_test_split(data, labels, test_size=0.40)
data_val, data_test, labels_val, labels_test = train_test_split(data_rest, labels_rest, test_size=0.625)


# Model Template
model = Sequential() # declare model

model.add(Dense(25, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
model.add(Dense(50, activation='tanh', kernel_initializer='he_normal'))
model.add(Dense(50, activation='selu', kernel_initializer='he_normal'))
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(data_train, labels_train, 
                    validation_data = (data_val, labels_val), 
                    epochs=50, 
                    batch_size=512)


# Report Results
print(history.history)
model_eval = model.evaluate(data_val, labels_val, batch_size=512)
print("")
print("")
print("************************************************************************")
print("[===============Result=================]")
print("Accuracy of the model is: %.2f%%" % (model_eval[1]*100))
prediction = model.predict(data_test)
print("[======================================]")

# Plot a Graph showing training set and validation accuracy over epoch
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

image_count = 0
i = 0
error = [0,0,0]
pTemp = [0,0,0,0,0,0,0,0,0,0]

# Get Wrongly Classified Image

# Iterate through every element in the test set
while i < 1625:
    j = 0
    for p in prediction[i]:
        pTemp[j] = round(p)
        j+=1
    prediction[i] = pTemp
    a = 0
    tTemp = labels_test[i]
    if image_count < 3:
        for predicted in prediction[i]:
            if tTemp[a] != predicted:
                error[image_count] = i
                image_count += 1
                break
            a += 1    
    i+=1

#Construction of the confusion matrix
size = 0
confusion_matrix = [[0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0]]

while size < 1625:
    preMarker = 0
    testMarker = 0
    for test in labels_test[size]:
        if test == 1:
            for pre in prediction[size]:
                if pre == 1:
                    temp = confusion_matrix[testMarker][preMarker]
                    confusion_matrix[testMarker][preMarker] = temp + 1
                if preMarker < 9:
                    preMarker += 1
        if testMarker < 9:
            testMarker += 1
    size += 1

#Print out the confusion matrix
print("")
print("[============Confusion Matrix===========]")
print("Prediction --------> Row")
print("True --------> Column")
print("")
print(np.matrix(confusion_matrix))
print("[=======================================]")
print("************************************************************************")


# Visulize 3 missclassified images
for e in error:
    label = labels_test
    pixels = data_test[e]
    pixels = np.array(pixels, dtype='uint8')
    reshaped_pixels = pixels.reshape(28,28)
    plt.title(prediction[e])
    plt.imshow(reshaped_pixels, cmap= 'gray')
    plt.show()
