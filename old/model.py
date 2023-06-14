import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

class_names = ["Damage/Dirt", "No Damage/Dirt"]


train_image="C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\train\\images"
test_image="C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\test\\images"
train_label="C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\train\\labels"
test_label="C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\test\\labels"

# Input shape (height, width, and number of channels of the image) they are all (371,586,3)
image_path = "C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\train\\images\\DJI_0004_03_07.png"

image = cv2.imread(image_path)

image_shape = image.shape

image_height, image_width, num_channels = image_shape

# Load image data
train_images = []
test_images = []

limit = 1000

for image_path in os.listdir(train_image)[:limit]:
    image = cv2.imread(train_image+"\\"+ image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = image / 255.0  # Scale pixel values to [0, 1]
    train_images.append(image)

for image_path in os.listdir(test_image)[:limit]:
    image = cv2.imread(test_image+"\\"+ image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = image / 255.0  # Scale pixel values to [0, 1]
    test_images.append(image)

train_images = np.array(train_images)
test_images = np.array(test_images)

train_labels = []
test_labels = []

for text_file in os.listdir(train_label)[:limit]:
    with open(train_label  + "\\"+ text_file, 'r') as f:
        train_labels.append(f.read().strip())

for text_file in os.listdir(test_label)[:limit]:
    with open(test_label  + "\\" + text_file, 'r') as f:
        test_labels.append(f.read().strip())

train_labels = np.array(train_labels)
train_labels = train_labels == class_names[0] 

test_labels = np.array(test_labels)
test_labels = test_labels == class_names[0]

# # Load label data
# train_labels = np.loadtxt(train_label, dtype=int)
# test_labels = np.loadtxt(test_label, dtype=int)

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names)))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy','f1-score'])

# train the model
history = model.fit(train_images, train_labels, epochs=30, 
                    validation_data=(test_images, test_labels), batch_size=35)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("test accuracy:", test_acc)


