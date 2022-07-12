from re import VERBOSE
import typing
from abc import abstractmethod
import tempfile

from tqdm import tqdm
import numpy as np
import mlflow
import pandas as pd
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

from .base import Metric
from eval_metrics import EvaluationMetrics


class Evaluator:

    def __init__(self, eval_metrics: typing.Optional[typing.List[Metric]]):
        self.eval_metrics_ = eval_metrics

    def run(self,
            tf_model: tfk.Model,
            data_loader: tf.data.Dataset,
            n_iter: int,
            active_run: mlflow.ActiveRun):
        # def _wrap_pred_step(model):
        #     """Overrides the ``predict`` method of the ``tfk.Model`` model.

        #     By calling ``predict`` method of the model, three lists will be returned:
        #      ``(predictions, ground truths, data_ids/sample_weights)``
        #     """

        #     def new_predict_step(data):
        #         x, y, z = tfk.utils.unpack_x_y_sample_weight(data)
        #         return model(x, training=False), y, z

        #     setattr(model, 'predict_step', new_predict_step)

        # def wrapper_gen(gen):
        #     while True:
        #         x_b, y_b, w_b = next(gen)
        #         yield x_b, y_b

        # # Using model.evaluate()
        # data_gen = iter(wrapper_gen(iter(data_loader)))
        # eval_internal_metrics = dict()
        # for k, v in tf_model.evaluate(data_gen,
        #                               steps=n_iter,
        #                               return_dict=True,
        #                               verbose=False).items():
        #     eval_internal_metrics[f'model.evaluate_{k}'] = v

        y_hat = tf_model.predict(data_loader, steps = n_iter)
        y_true = data_loader.get_y_true()
        y_true_single = np.argmax(y_true, axis=1)
        y_pred_single = np.argmax(y_hat, axis=1)
        y_pred = np.array([list(np.eye(2)[i]) for i in y_pred_single], dtype=np.uint8)

        performance = EvaluationMetrics()
        class_names = ['Non-Hemorrhage', 'Hemorrhage']
        evaluated = performance.evaluate_classification_report(y_true_single, y_pred_single, class_names)

        metrics_dict = {'Classification Report': evaluated}
        self._log_metrics_to_mlflow(active_run, metrics_dict)

        # # Using self.get_metrics()
        # metrics = [metric for metric in self.eval_metrics_]
        # if any(metrics):
        #     _wrap_pred_step(tf_model)
        #     preds, gts, data_ids = tf_model.predict(data_loader, steps=n_iter)
        #     report_df = self._generate_eval_reports(metrics, preds, gts, data_ids)
        #     self._log_df_report_to_mlflow(active_run, report_df)

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
            #
            # test_metrics = {}
            # for c in summary_report.columns:
            #     metric_name = f'{prefix}_{c}'
            #     metric_value = summary_report[c]['mean']
            #     test_metrics[metric_name] = metric_value
            #
            # mlflow.log_metrics(test_metrics)
