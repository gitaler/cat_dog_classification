from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, List, Tuple
from tensorflow.keras.models import Model as KerasModel


def model_evaluation(model: KerasModel, data_generator: ImageDataGenerator) \
        -> Tuple[float, float, float, float, List[List[int]]]:
    """
    calculates evaluation metrics given a model and a data generator
    :param model:
    :param data_generator:
    :return: accuracy, precision, recall, f1-score, confusion matrix
    """
    total_predictions = []
    total_labels = []
    for i, batch in enumerate(data_generator):
        images, labels = batch
        preds = model.predict(images, verbose=0)
        preds = np.where(preds < 0.5, 0, 1)
        preds = list(map(lambda x: x[0], preds))
        total_predictions = total_predictions + preds
        total_labels = total_labels + list(labels)
        if i == len(data_generator) - 1:
            break

    acc = accuracy_score(total_labels, total_predictions)
    prec = precision_score(total_labels, total_predictions)
    rec = recall_score(total_labels, total_predictions)
    f1 = f1_score(total_labels, total_predictions)
    conf_mat = confusion_matrix(total_labels, total_predictions)

    return acc, prec, rec, f1, conf_mat


def plot_training_curves(loss_evolution_dict: Dict[str, List[float]]) -> None:
    """
    plots training losses
    :param loss_evolution_dict:
    :return: None
    """
    plt.figure(figsize=(15, 8))
    plt.plot(loss_evolution_dict['loss'], label='Train Loss')
    plt.plot(loss_evolution_dict['val_loss'], label='Validation Loss')
    plt.plot(loss_evolution_dict['accuracy'], label='Train Accuracy')
    plt.plot(loss_evolution_dict['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()
