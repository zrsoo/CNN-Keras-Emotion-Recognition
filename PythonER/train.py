from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

data_train = ImageDataGenerator(rescale=1./255)
data_validation = ImageDataGenerator(rescale=1./255)

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