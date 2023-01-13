import numpy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(numpy.arange(1, len(model_history.history['accuracy']) + 1),
                      len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(numpy.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()


data_train = ImageDataGenerator(rescale=1. / 255)
data_validation = ImageDataGenerator(rescale=1. / 255)

# Preprocess test images
train_gen = data_train.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical')

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
