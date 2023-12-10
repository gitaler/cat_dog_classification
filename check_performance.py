import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import pandas as pd
from evaluation import model_evaluation

# read test cats file names
with open('./results/cats_test.pickle', 'rb') as file_pkl:
    test_cats = pickle.load(file_pkl)
test_cats = list(map(lambda x: './dataset/PetImages/Cat/'+x, test_cats))

# read test dogs file names
with open('./results/dogs_test.pickle', 'rb') as file_pkl:
    test_dogs = pickle.load(file_pkl)
test_dogs = list(map(lambda x: './dataset/PetImages/Dog/'+x, test_dogs))


# combine cats and dogs data
combined_data = [(path, '0') for path in test_cats] + [(path, '1') for path in test_dogs]
image_paths = [data[0] for data in combined_data]
labels = [data[1] for data in combined_data]

# prepare data generator from dict
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({"image_path": image_paths, "label": labels}),
    x_col="image_path",
    y_col="label",
    target_size=(128, 128),
    batch_size=64,
    class_mode='binary'
)

# load the trained model and evaluate it
model = load_model('./results/best_classifier.h5', compile=False)
acc, prec, rec, f1, conf_mat = model_evaluation(model, generator)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)
print("Confusion Matrix:\n", conf_mat)
