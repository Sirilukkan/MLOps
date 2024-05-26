import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, load_pipeline





def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    precision = metrics.precision_score(actual, pred, pos_label=1)
    recall = metrics.recall_score(actual, pred, pos_label=1)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    
    return accuracy, precision, recall, f1, auc, fpr, tpr

def plot_roc_curve(fpr, tpr, auc, model_key):
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve area = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')

    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/ROC_curve_{model_key}.png")
    # Close plot
    plt.close()

def plot_confusion_matrix(actual, pred, model_key):
    cm = metrics.confusion_matrix(actual, pred)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]:d}", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/Confusion_Matrix_{model_key}.png")
    # Close plot
    plt.close()

def evaluate_models():
    # Load training dataset
    train_data = load_dataset(config.TRAIN_FILE)
    X_train = train_data[config.FEATURES]
    y_train = train_data[config.TARGET].map({'Positive': 1, 'Negative ': 0})  # Note: fix space issue in 'Negative'
    
    # Load and evaluate each model
    for model_key in config.MODEL_NAMES.keys():
        model_pipeline = load_pipeline(model_key)
        if model_pipeline:
            y_pred = model_pipeline.predict(X_train)
            accuracy, precision, recall, f1, auc, fpr, tpr = eval_metrics(y_train, y_pred)
            print(f"Evaluation metrics for {model_key}:")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print(f"AUC: {auc}\n")

            # Plot ROC curve
            plot_roc_curve(fpr, tpr, auc, model_key)

            # Plot Confusion Matrix
            plot_confusion_matrix(y_train, y_pred, model_key)
        else:
            print(f"Model pipeline for '{model_key}' could not be loaded.")

if __name__ == '__main__':
    evaluate_models()
