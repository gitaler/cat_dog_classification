from data_utils import *
from generators import *
from model import *
from evaluation import *
import os


def main():
    print('--- Dataset download and stats ---')
    # dataset download and unzip
    # URL = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
    # create_dataset(URL)

    # print useful dataset stats
    cats_folder = './dataset/PetImages/Cat/'
    dogs_folder = './dataset/PetImages/Dog/'
    cats, cats_resolutions, cats_not_opened_imgs = get_images_stats(cats_folder)
    dogs, dogs_resolutions, dogs_not_opened_imgs = get_images_stats(dogs_folder)
    remove_corrupted_images(cats_not_opened_imgs + dogs_not_opened_imgs)
    avg_resolution = average_resolution({**cats_resolutions, **dogs_resolutions})
    print('average images resolution', avg_resolution)
    print('# cats:', len(cats))
    print('# dogs:', len(dogs))
    print('removed cats files:', cats_not_opened_imgs)
    print('removed dogs files:', dogs_not_opened_imgs)

    # train and test directories creation
    train_folder = './dataset/train/'
    test_folder = './dataset/test/'
    create_train_test_directories(train_folder, test_folder)

    # train and test sets split
    print('\n--- Splitting images in Train and Test sets ---')
    train_percentage = 85 / 100
    cats_fails = train_test_split(cats, train_percentage, train_folder + 'cats/', test_folder + 'cats/')
    dogs_fails = train_test_split(dogs, train_percentage, train_folder + 'dogs/', test_folder + 'dogs/')

    # image data generators
    print('\n--- Image Data Generators ---')
    BATCH_SIZE = 64
    RESHAPE_RES = (128, 128)  # reshaped image size
    train_gen, val_gen = create_train_and_val_generator(train_folder, BATCH_SIZE, RESHAPE_RES)
    test_gen = create_test_generator(test_folder, BATCH_SIZE, RESHAPE_RES)

    # model creation and training
    print('\n--- Model training ---')
    EPOCHS = 100
    batch_images, batch_labels = train_gen.next()
    input_shape = batch_images[0].shape
    model = create_model(input_shape)
    model, history = train_model(model, train_gen, val_gen, EPOCHS)
    os.makedirs('./trained model/', exist_ok=True)
    model.save('./trained model/classifier.h5')

    # model evaluation
    print('\n--- Model evaluation metrics ---')
    plot_training_curves(history.history)
    acc, prec, rec, f1, conf_mat = model_evaluation(model, test_gen)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", conf_mat)


if __name__ == "__main__":
    main()
    # exec(open("check_performance.py").read())
