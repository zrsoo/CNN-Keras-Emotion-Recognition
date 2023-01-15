import numpy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf


# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 2.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.show()


data_train = ImageDataGenerator(rescale=1. / 255, rotation_range=0.1, horizontal_flip=True)
data_validation = ImageDataGenerator(rescale=1. / 255, rotation_range=0.1, horizontal_flip=True)

# Preprocess test images
train_gen = data_train.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    )

# Preprocess train images
validation_gen = data_validation.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical')

# Create CNN model
emodel = Sequential()

emodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emodel.add(MaxPooling2D(pool_size=(2, 2)))
emodel.add(Dropout(0.25))

emodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emodel.add(MaxPooling2D(pool_size=(2, 2)))
emodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emodel.add(MaxPooling2D(pool_size=(2, 2)))
emodel.add(Dropout(0.25))

emodel.add(Flatten())
emodel.add(Dense(1024, activation='relu'))
emodel.add(Dropout(0.5))
emodel.add(Dense(7, activation='softmax'))


emodel.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

# Train model
emodel_info = emodel.fit(
    train_gen,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_gen,
    validation_steps=7178 // 64)

# Save to json
json_model = emodel.to_json()
with open("emodel.json", "w") as file:
    file.write(json_model)

# Save weights in h5 file
emodel.save_weights('emodel.h5')

plot_model_history(emodel_info)

