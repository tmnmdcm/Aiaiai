import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

class_names = ["Damage/Dirt", "No Damage/Dirt"]

# Specify the desired image size
target_image_size = (128, 128)

train_image = "C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\train\\images"
test_image = "C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\test\\images"
train_label = "C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\train\\labels"
test_label = "C:\\Users\\tomas\\OneDrive\\Documentos\\TU Delft 2022-2023\\DSE\\AI Model\\dataset\\test\\labels"

# Load image data
train_images = []
test_images = []

limit = 1000

for image_path in os.listdir(train_image)[:limit]:
    image = cv2.imread(os.path.join(train_image, image_path))
    image = cv2.resize(image, target_image_size)
    image = image / 255.0  # Scale pixel values to [0, 1]
    train_images.append(image)

for image_path in os.listdir(test_image)[:limit]:
    image = cv2.imread(os.path.join(test_image, image_path))
    image = cv2.resize(image, target_image_size)
    image = image / 255.0  # Scale pixel values to [0, 1]
    test_images.append(image)

train_images = np.array(train_images)
test_images = np.array(test_images)

train_labels = []
test_labels = []

for text_file in os.listdir(train_label)[:limit]:
    with open(os.path.join(train_label, text_file), 'r') as f:
        train_labels.append(f.read().strip())

for text_file in os.listdir(test_label)[:limit]:
    with open(os.path.join(test_label, text_file), 'r') as f:
        test_labels.append(f.read().strip())

train_labels = np.array(train_labels)
train_labels = train_labels == class_names[0]

test_labels = np.array(test_labels)
test_labels = test_labels == class_names[0]

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=target_image_size + (3,)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Use a single unit with sigmoid activation for binary classification

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', 'Precision', 'Recall'])

# train the model
history = model.fit(train_images, train_labels, epochs=5, 
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
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_images, test_labels, verbose=2)
test_f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
print("test accuracy:", test_acc)
print("test precision:", test_precision)
print("test recall:", test_recall)
print("test F1-score:", test_f1_score)