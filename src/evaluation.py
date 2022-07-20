import typing
import tempfile

from tqdm import tqdm
import numpy as np
import mlflow
import pandas as pd
import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers

from .base import Metric
from src.eval_metrics import EvaluationMetrics


class Evaluator:

    def __init__(self, eval_metrics: typing.Optional[typing.List[Metric]]):
        self.eval_metrics_ = eval_metrics

    def run(self,
            tf_model: tfk.Model,
            data_loader: tf.data.Dataset,
            n_iter: int,
            active_run: mlflow.ActiveRun):

        performance = EvaluationMetrics()

        y_hat = tf_model.predict(data_loader, steps = n_iter)
        y_pred = []
        for pred in y_hat:
            if pred >= 0.5:
                pred = 1
                y_pred.append(pred)
            else:
                pred = 0
                y_pred.append(pred)

        y_true = data_loader.get_y_true()
        x_true = data_loader.get_x_true()

        list_of_tuples = list(zip(x_true, y_true, y_pred))
        df = pd.DataFrame(list_of_tuples,columns=['ID', 'GT', 'Pred'])
        df["Status"] = ""

        condlist = [(df['GT'] == 1) & (df['Pred'] == df['GT']), (df['GT'] == 0) & (df['Pred'] == df['GT']), (df['GT'] == 1) & (df['Pred'] != df['GT']), (df['GT'] == 0) & (df['Pred'] != df['GT'])]
        choicelist = ['True Positive', 'True Negative', 'False Negative', 'Fasle positive']
        df['Status'] = np.select(condlist, choicelist)

        evaluated_acc = performance.evaluate_accuracy(y_true, y_pred)
        evaluated_precision_score = performance.evaluate_precision_score(y_true, y_pred)
        evaluated_recall_score = performance.evaluate_recall_score(y_true, y_pred)
        evaluated_f1_score = performance.evaluate_f1_score(y_true, y_pred)
        evaluated_roc_auc_score = performance.evaluate_roc_auc_score(y_true, y_pred)
        evaluated_sensitivity = performance.evaluate_sensitivity(y_true, y_pred)
        evaluated_specifity = performance.evaluate_specifity(y_true, y_pred)
        evaluated_npv = performance.evaluate_npv(y_true, y_pred)

        metrics_dict = {'Acc_Score': evaluated_acc, 'Precision_Score': evaluated_precision_score,
                        'Recall_Score': evaluated_recall_score, 'F1_score': evaluated_f1_score,
                        'Roc_Auc': evaluated_roc_auc_score, 'Sensitivity': evaluated_sensitivity,
                        'Specifity': evaluated_specifity, 'NPV': evaluated_npv}

        self._log_metrics_to_mlflow(active_run, metrics_dict)
        self._log_df_report_to_mlflow(active_run, df)

    @staticmethod
    def _generate_eval_reports(metrics: typing.List[Metric],
                               preds: np.ndarray,
                               gts: np.ndarray,
                               data_ids: np.ndarray) -> typing.Optional[pd.DataFrame]:
        eval_func_dict = {metric.name: metric.get_func() for metric in metrics}
        report = {metric.name: list() for metric in metrics}

        n_data = len(preds)
        indxs = list()
        with tqdm(total=n_data) as pbar:
            for ind, (y_pred, y_true, data_id) in enumerate(zip(preds, gts, data_ids)):
                for k, v in report.items():
                    metric_val = eval_func_dict[k](y_true, y_pred)
                    if isinstance(metric_val, tf.Tensor):
                        v.append(metric_val.numpy())
                    else:
                        v.append(metric_val)

                if np.array(data_id).ndim > 0:
                    d_id = data_id[0]
                else:
                    d_id = data_id
                if isinstance(d_id, tf.Tensor):
                    d_id = d_id.numpy()
                    if isinstance(d_id, bytes):
                        indxs.append(d_id.decode())
                    else:
                        indxs.append(d_id)
                else:
                    indxs.append(str(d_id))

                pbar.update(1)

        df = pd.DataFrame(report, index=indxs)
        return df

    @staticmethod
    def _log_metrics_to_mlflow(active_run: typing.Optional[mlflow.ActiveRun],
                               metrics: dict):
        if active_run is not None:

            metrics_to_log = {}
            for k, v in metrics.items():
                metric_name = f'{k}'
                metric_value = v
                metrics_to_log[metric_name] = metric_value
            mlflow.log_metrics(metrics_to_log)

    @staticmethod
    def _log_df_report_to_mlflow(active_run: typing.Optional[mlflow.ActiveRun],
                                 report_df: pd.DataFrame):
        if active_run is not None:
            with tempfile.NamedTemporaryFile(prefix='eval-report-', suffix='.csv') as f:
                report_df.to_csv(f.name)
                mlflow.log_artifact(f.name)

            summary_report = report_df.describe()
            with tempfile.NamedTemporaryFile(prefix='eval-report-summary-', suffix='.csv') as f:
                summary_report.to_csv(f.name)
                mlflow.log_artifact(f.name)
