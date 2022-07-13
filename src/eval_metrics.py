import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn import metrics


def print_confusion_matrix(y_test, y_pred, class_names):
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, cmap='crest', linecolor='white', linewidths=1, annot=True, xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


class EvaluationMetrics:

    @staticmethod
    def evaluate_accuracy(y_test, y_pred):
        return np.round(metrics.accuracy_score(y_test, y_pred), 4)

    @staticmethod
    def evaluate_precision_score(y_test, y_pred):
        return np.round(metrics.precision_score(y_test, y_pred), 4)

    @staticmethod
    def evaluate_recall_score(y_test, y_pred):
        return np.round(metrics.recall_score(y_test, y_pred), 4)

    @staticmethod
    def evaluate_f1_score(y_test, y_pred):
        return np.round(metrics.f1_score(y_test, y_pred), 4)

    @staticmethod
    def evaluate_roc_auc_score(y_test, y_pred):
        return metrics.roc_auc_score(y_test, y_pred)

    @staticmethod
    def evaluate_classification_report(y_test, y_pred, target_names):
        print(target_names)
        return metrics.classification_report(y_test, y_pred, target_names=target_names)

