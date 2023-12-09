from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple


def create_train_and_val_generator(train_dir: str, batch_size: int = 64,
                                   img_target_resolution: Tuple[int, int] = (128, 128)) \
        -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    returns train and validation data generators. images transformations are applied
    :param train_dir: train folder path
    :param batch_size:
    :param img_target_resolution: reshape images to specified resolution
    :return: train and val data generators
    """
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.15
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_target_resolution,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_target_resolution,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator


def create_test_generator(test_dir: str, batch_size: int = 64,
                          img_target_resolution: Tuple[int, int] = (128, 128)) -> ImageDataGenerator:
    """
    returns test data generator
    :param test_dir: test folder path
    :param batch_size:
    :param img_target_resolution:
    :return: test data generator
    """
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_target_resolution,
        batch_size=batch_size,
        class_mode='binary',
    )

    return test_generator
