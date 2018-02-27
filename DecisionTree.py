import data_service_DT as ds
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from skimage import io, img_as_float
from PIL import Image



# 1) baseline model
x_train = ds.matrix_image_training # use 60%
y_train = ds.matrix_label_training # use 60%
x_test = ds.matrix_image_test 
y_test = ds.matrix_label_test
x_validation = ds.matrix_image_validation
y_validation = ds.matrix_label_validation
DT_Model = DecisionTreeClassifier(random_state = 0) 
history = DT_Model.fit(x_train, y_train)

print('Training set accuracy: {:.3f}'.format(DT_Model.score(x_train, y_train)))
print('Validation set accuracy: {:.3f}'.format(DT_Model.score(x_validation, y_validation)))
print('Testing set accuracy: {:.3f}'.format(DT_Model.score(x_test, y_test)))


# 2) variation model
max_depth = 10
variation_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
history2 = variation_tree.fit(x_train, y_train)
print('\nmax_depth = ', max_depth)
print('Training set accuracy: {:.3f}'.format(variation_tree.score(x_train, y_train)))
print('Validation set accuracy: {:.3f}'.format(variation_tree.score(x_validation, y_validation)))
print('Testing set accuracy: {:.3f}'.format(variation_tree.score(x_test, y_test)))


# 3) new features

# 1st feature: average intensity of pixels of the whole image
def avg_pixel(images):
	avg_pixel_val = []
	for i in range(0, len(images)):
		image = images[i]
		image = img_as_float(image)
		avg_pixel_val.append(np.mean(image))
	return avg_pixel_val

# average intensity of pixels of left half split image (14 * 28)
def avg_pixel_2(images):
	avg_pixel_2_val = []
	for i in range(0, len(images)):
		img = images[i]
		#print(img.shape)
		#img = img[0:14, 0:27]
		float_img = img_as_float(img)
		avg_pixel_2_val.append(np.mean(float_img))
	return avg_pixel_2_val


# average intensity of pixels of left up quaterly split image (14 * 14)
def avg_pixel_4(images):
	avg_pixel_4_val = []
	for i in range(0, len(images)):
		img = images[i]
		img = img[0:14, 0:14]
		float_img = img_as_float(img)
		avg_pixel_4_val.append(np.mean(float_img))
	return avg_pixel_4_val

# average intensity of pixels of left up of 8 peices splited (7 * 7)
def avg_pixel_8(images):
	avg_pixel_8_val = []
	for i in range(0, len(images)):
		img = images[i]
		#print(img.shape)
		img = img[0:7, 0:7]
		float_img = img_as_float(img)
		avg_pixel_8_val.append(np.mean(float_img))
	return avg_pixel_8_val


# extract
def extract_train(images):
	extract_list = []
	avg_pixel_val = avg_pixel(images)
	avg_pixel_2_val = avg_pixel_2(images)
	avg_pixel_4_val = avg_pixel_4(images)
	avg_pixel_8_val = avg_pixel_8(images)
	length = len(images)* 0.6
	for i in range(0, int(length)):
		extract_array = np.array([avg_pixel_val[i], avg_pixel_2_val[i], avg_pixel_4_val[i], avg_pixel_8_val[i]])
		extract_list.append(extract_array)

	return extract_list

def extract_test(images):
	extract_list = []
	avg_pixel_val = avg_pixel(images)
	avg_pixel_2_val = avg_pixel_2(images)
	avg_pixel_4_val = avg_pixel_4(images)
	avg_pixel_8_val = avg_pixel_8(images)
	for j in range(int(len(images)* 0.75), len(images)):
		extract_array = np.array([avg_pixel_val[j], avg_pixel_2_val[j], avg_pixel_4_val[j], avg_pixel_8_val[j]])
		extract_list.append(extract_array)
	return extract_list


extract_Model = DecisionTreeClassifier(random_state = 0) 


raw_images = np.load('images.npy')
# convert image to vector
images = []
for raw_image in raw_images:
    image = raw_image.reshape(28, 28)
    images.append(image)

print(len(extract_train(images)))
lables = []
for i in range(0, 3900):
	lable = y_train[i]
	lables.append(lable)
print(len(lables))

extract_history = extract_Model.fit(extract_train(images), lables)

print("Extract DT Accuracy Score:", extract_Model.score(extract_train(images), lables))

"""
for i in range(4875, 6500):
	lable_test.append() 
	"""
print("test")
print(len(images))
print(len(extract_test(images)))
raw_labels = np.load('labels.npy')
test_lables = []
for i in range(int(len(images)* 0.75), len(images)):
	a_lable = raw_labels[i]
	test_lables.append(a_lable)

print(len(y_test))
print("Extract DT Accuracy Score:", extract_Model.score(extract_test(images), test_lables))


# construct confusion matrix
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

confusion_matrix2 = confusion_matrix
confusion_matrix3 = confusion_matrix
print("#1 confusion matrix")
validation_predict = DT_Model.predict(x_validation)

for i in range(0, len(validation_predict)):
    confusion_matrix[y_validation[i]][validation_predict[i]] += 1

for i in range(0, len(confusion_matrix)):
	print(confusion_matrix[i])

print("\n#2 confusion matrix")

variation = variation_tree.predict(x_validation)


for i in range(0, len(variation)):
    confusion_matrix2[y_validation[i]][variation[i]] += 1

for i in range(0, len(confusion_matrix)):
	print(confusion_matrix2[i])

print("\n#3 confusion matrix")
validation_set = []
for i in range(3900, 4875):
	validation_ex = images[i]
	validation_set.append(validation_ex)

feature_predict = extract_Model.predict(extract_test(images))

for i in range(0, len(variation)):
    confusion_matrix3[y_validation[i]][variation[i]] += 1

for i in range(0, len(confusion_matrix)):
	print(confusion_matrix3[i])



