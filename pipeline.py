from pathlib import Path
import pickle
from pydoc import locate
import inspect

import mlflow
from omegaconf import OmegaConf
from git import Repo
import pandas as pd
from metaflow import FlowSpec, step, Parameter
import tensorflow as tf

tfk = tf.keras

from src.utils import check_conditions, setup_mlflow_active_run, get_mlflow_run
from src.data_ingestion import prepare_data
from src.training import Trainer, Exporter, get_checkpoints_info
from src.evaluation import Evaluator
from src.inference import MyHemoModel
from src.base import Metric
import src.eval_metrics
from src import DATA_FOLDER


class Pipeline(FlowSpec):
    config_path: str = Parameter('config-path',
                                 default='config.yaml',
                                 help='path to the config file')
    meta_data_path: str = Parameter('meta-data',
                                    help='path to meta-data')
    output_dir: str = Parameter('output-dir',
                                help='where to write the prepared data')
    do_prepare: bool = Parameter('prepare',
                                 is_flag=True,
                                 help='whether to execute the preparation step.')
    do_train: bool = Parameter('train',
                               is_flag=True,
                               help='whether to execute the training step.')
    do_evaluation: bool = Parameter('evaluate',
                                    is_flag=True,
                                    help='whether to execute the evaluation step.')
    do_export: bool = Parameter('export',
                                is_flag=True,
                                help='whether to execute the export step.')
    colab: bool = Parameter('colab',
                            is_flag=True,
                            default=False,
                            help='set this to true if you are running this pipeline in google colab environment.')
    debug: bool = Parameter('debug',
                            is_flag=True,
                            default=False,
                            help='set this to true, to skip the commit-check of the code.')

    root_data_dir = DATA_FOLDER
    run_dir = Path('run')
    run_info_path = Path('.run.pkl')

    @step
    def start(self):
        """Starting the pipeline"""

        self.data_dir = self.root_data_dir.joinpath('hemo')
        self.meta_data_df = pd.read_csv(self.meta_data_path)
        config_file_path = Path(self.config_path)
        self.config = OmegaConf.load(config_file_path)
        print(f'loaded config file {self.config_path}')

        self.run_dir.mkdir(exist_ok=True)

        if Path(self.run_info_path).exists():
            with open(self.run_info_path, 'rb') as f:
                run_dict = pickle.load(f)
            self.active_run = get_mlflow_run(
                    parent_exp_id=run_dict['experiment_id'],
                    parent_run_id=run_dict['run_id'],
                    colab=self.colab)
        else:
            self.active_run = None

        self.next(self.check_conditions)

    @step
    def check_conditions(self):
        """check whether the data and code are committed."""

        unique_datasources = self.meta_data_df['DataSource'].unique().tolist()

        self.repo = Repo('.')

        if not self.debug:
            for ds in unique_datasources:
                check_conditions(self.repo, self.root_data_dir.joinpath(ds))
            check_conditions(self.repo, self.data_dir)

        self.next(self.prepare)

    @step
    def prepare(self):
        """prepare (ingest) the data to be ready for loading with DataLoader"""

        if self.do_prepare:
            prepare_data(self.meta_data_df, Path(self.output_dir))

        self.next(self.train)

    @step
    def train(self):
        """Train the model and log to MLFlow and Discord"""

        if self.do_train:
            print(f'instantiating {self.config.dataset} and config.model ..')
            dataset = locate(self.config.dataset)(self.config, self.data_dir)
            model_builder = locate(self.config.model)(self.config)
            trainer = Trainer(self.config, self.run_dir)

            tr_gen, n_tr, val_gen, n_val = dataset.create_train_val_generators()

            root = Path(self.repo.working_tree_dir).name
            experiment_name = root + '/' + self.repo.active_branch.name

            self.active_run = setup_mlflow_active_run(config_path=Path(self.config_path),
                                                      session_type='training',
                                                      experiment_name=experiment_name,
                                                      colab=self.colab)
            with open(self.run_info_path, 'wb') as f:
                pickle.dump(dict(self.active_run.info), f)

            trainer.train(model_builder=model_builder,
                          train_data_gen=tr_gen,
                          n_iter_train=n_tr,
                          val_data_gen=val_gen,
                          n_iter_val=n_val,
                          active_run=self.active_run)

        # if self.do_evaluation:
        #     self.next(self.evaluate)
        # elif self.do_export:
        #     self.next(self.export)
        # else:
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate the best model."""

        if self.do_evaluation:
            # dataset = Dataset(self.config, self.data_dir)
            dataset = locate(self.config.dataset)(self.config, self.data_dir)
            data_loader, n_iter = dataset.create_eval_generator()

            checkpoints = get_checkpoints_info(self.run_dir.joinpath('checkpoints'))
            if self.config.trainer.export_mode == 'min':
                selected_model = min(checkpoints, key=lambda x: x['value'])
            else:
                selected_model = max(checkpoints, key=lambda x: x['value'])

            model = tfk.models.load_model(selected_model['path'])

            eval_metrics = list()
            for name, cls in inspect.getmembers(src.eval_metrics,
                                                lambda o: inspect.isclass(o) and issubclass(o, Metric)):
                if name != 'Metric':
                    eval_metrics.append(cls())

            evaluator = Evaluator(eval_metrics=eval_metrics)

            active_run = get_mlflow_run(parent_exp_id=self.active_run.info.experiment_id,
                                        parent_run_id=self.active_run.info.run_id,
                                        colab=self.colab)
            with active_run:
                with mlflow.start_run(experiment_id=active_run.info.experiment_id,
                                      run_name='child-run-evaluate',
                                      nested=True) as nested_run:
                    print(f'artifact uri for evaluation nested run {nested_run.info.artifact_uri}')
                    mlflow.log_artifact(self.config_path)
                    mlflow.set_tag("session_type", 'evaluation')  # ['hpo', 'evaluation', 'training', 'export']
                    evaluator.run(model, data_loader, n_iter, nested_run)

        # if self.do_export:
        #     self.next(self.export)
        # else:
        self.next(self.export)

    @step
    def export(self):
        """Export the best model as an artifact to MLFlow"""

        if self.do_export:
            exporter = Exporter(config=self.config, run_dir=self.run_dir)

            pyfunc_model = MyHemoModel()

            active_run = get_mlflow_run(parent_exp_id=self.active_run.info.experiment_id,
                                        parent_run_id=self.active_run.info.run_id,
                                        colab=self.colab)
            with active_run:
                with mlflow.start_run(experiment_id=active_run.info.experiment_id,
                                      run_name="child-run-export",
                                      nested=True) as nested_run:
                    # mlflow.log_artifact(self.config_path)
                    mlflow.set_tag("session_type", 'export')  # ['hpo', 'evaluation', 'training', 'export']
                    exporter.log_model_to_mlflow(active_run=nested_run,
                                                 pyfunc_model=pyfunc_model,
                                                 config_path=Path(self.config_path))

        self.next(self.end)

    @step
    def end(self):
        print('done.')


if __name__ == '__main__':
    Pipeline()
