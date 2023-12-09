from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import History as KerasHistory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple


def create_model(input_shape: Tuple[int, int, int]) -> KerasModel:
    """
    creates and return the classifier CNN model
    :param input_shape: images shape
    :return: compiled model
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model: KerasModel, train_generator: ImageDataGenerator,
                validation_generator: ImageDataGenerator, epochs: int) \
        -> Tuple[KerasModel, KerasHistory]:
    """
    model training. early stopping(val_loss) is specified to avoid overfitting
    :param model: compiled model
    :param train_generator: train data generator
    :param validation_generator: val data generator
    :param epochs: number of maximum training iterations
    :return: trained model, training history(losses curves)
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[early_stopping]
    )
    return model, history
