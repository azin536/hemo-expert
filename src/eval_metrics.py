import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn import metrics

def print_confusion_matrix(y_test, y_pred, class_names):
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm,cmap='crest',linecolor='white',linewidths=1,annot=True, xticklabels = class_names, yticklabels = class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

class Performance:
  @staticmethod
  def print_performance_metrics(y_test, y_pred, class_names):
      # print('Accuracy:', np.round(metrics.accuracy_score(y_test, y_pred),4))
      # print('Precision:', np.round(metrics.precision_score(y_test, y_pred),4))
      # print('Recall:', np.round(metrics.recall_score(y_test, y_pred),4))
      # print('F1 Score:', np.round(metrics.f1_score(y_test, y_pred),4))
      # if len(np.unique(y_test)) == 2:
      #     print('ROC AUC:',metrics.roc_auc_score(y_test,y_pred))
      # print('\t\tClassification Report:\n', metrics.classification_report(y_test, y_pred, target_name=class_names))
      accuracy_score = np.round(metrics.accuracy_score(y_test, y_pred),4)
      return accuracy_score
