from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np

model_file_name = 'classification_model.h5'
train_dataset_path = "dataset/training_set"
test_dataset_path = "dataset/test_set"
train_image_count = 8000
test_image_count = 2000
inputImageSize = (64, 64,)
convolutionSize = []
poolSize = (2, 2)
strides = (2, 2)
batch_size = 32
epochs = 25


def data_generator():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_datagen.flow_from_directory(train_dataset_path,
                                                     target_size=inputImageSize,
                                                     batch_size=batch_size,
                                                     class_mode="binary")
    test_set = test_datagen.flow_from_directory(test_dataset_path,
                                                target_size=inputImageSize,
                                                batch_size=batch_size,
                                                class_mode="binary")
    return training_set, test_set


def create_model():
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape=inputImageSize + (3,), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=poolSize, strides=strides))
    classifier.add(Convolution2D(64, 3, 3, activation="relu"))
    classifier.add(MaxPooling2D(pool_size=poolSize, strides=strides))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation="relu"))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=128, activation="relu"))
    classifier.add(Dense(units=1, activation="sigmoid"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    classifier.summary()
    return classifier


def load_image(path):
    picture = image.load_img(path, target_size=inputImageSize)
    picture = image.img_to_array(picture)
    return np.expand_dims(picture, axis=0)


model = create_model()
train_data, test_data = data_generator()
model.fit_generator(train_data,
                    samples_per_epoch=train_image_count,
                    epochs=epochs,
                    validation_data=test_data,
                    validation_steps=test_image_count)

model.save(model_file_name)


model = load_model(model_file_name)
test_image = load_image('dataset/single_prediction/d.jpg')
result = model.predict(test_image)
if result[0][0] == 1:
    print("It is classified as Dog.")
else:
    print("It is classified as Cat.")
