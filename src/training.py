from pathlib import Path
import typing
import os
import shutil

import tensorflow as tf
import tensorflow.keras.callbacks as tfkc
import tensorflow.keras as tfk
import mlflow
from omegaconf.dictconfig import DictConfig

from .model_building import ModelBuilderBase
# from .bot import DiscordBot


class Trainer:
    """Responsibilities:
        - training
        - generating tensorboard and mlflow training metrics
        - resume training if interrupted

    Attributes:

        - config: config file
        - run_dir: where to write the checkpoints

    Examples:

        >>> from pathlib import Path
        >>> from omegaconf import OmegaConf
        >>> from mlflow import ActiveRun
        >>> config = OmegaConf.load('./config.yaml')
        >>> run_dir = Path('run')
        >>> run_dir.mkdir(exist_ok=True)
        >>> exported_dir = Path('exported')
        >>> trainer = Trainer(config, run_dir, exported_dir)
        >>> trainer.train(...)


    """

    def __init__(self,
                 config: DictConfig,
                 run_dir: Path):

        self.config = config
        self.run_dir = run_dir

        # Paths
        self.checkpoints_dir = self.run_dir.joinpath('checkpoints')
        self.tensorboard_log_dir = self.run_dir.joinpath('logs')

        # Make logs and checkpoints directories
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
        self.tensorboard_log_dir.mkdir(exist_ok=True, parents=True)

        # Container for fit_history
        self.history_ = None

        # Discord Bot
        # self.bot = DiscordBot()

    def train(self,
              model_builder: ModelBuilderBase,
              train_data_gen: typing.Union[typing.Iterator, typing.Iterable],
              n_iter_train: int,
              val_data_gen: typing.Union[typing.Iterator, typing.Iterable],
              n_iter_val: int,
              active_run: typing.Optional[mlflow.ActiveRun] = None,
              retrain: bool = False):
        """Trains the model using data generators and logs to ``mlflow``.

        Will try to resume training from latest checkpoint, else starts training from ``epoch=0``.

        Args:
            model_builder: for building and compiling the model
            train_data_gen: preprocessed, augmented training-data-generator.
            n_iter_train: ``steps_per_epoch`` for ``model.fit``
            val_data_gen: preprocessed, augmented validation data-generator.
            n_iter_val: ``validation_steps`` for ``model.fit``
            active_run: ``mlflow.ActiveRun`` for logging to **mlflow**.
            retrain: if True, removes checkpoints and logs, and starts from scratch.

        """

        print(f'available GPU devices: {tf.config.list_physical_devices()}')

        model: tf.keras.Model

        if retrain or not any(_get_checkpoints(self.checkpoints_dir)):
            shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
            shutil.rmtree(self.tensorboard_log_dir, ignore_errors=True)

            self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
            self.tensorboard_log_dir.mkdir(exist_ok=True, parents=True)

            initial_epoch = 0
            model = model_builder.get_compiled_model()
        else:
            model, initial_epoch = self._load_latest_model()
            mlflow.set_tag('session_type', 'resumed_training')

        # Get callbacks
        callbacks = self._get_callbacks(model_builder, active_run)

        # Fit
        with active_run as _:
            history = model.fit(train_data_gen,
                                steps_per_epoch=n_iter_train,
                                initial_epoch=initial_epoch,
                                epochs=self.config.trainer.epochs,
                                validation_data=val_data_gen,
                                validation_steps=n_iter_val,
                                class_weight=model_builder.get_class_weight(),
                                callbacks=callbacks)
        self.history_ = history

    def _write_mlflow_run_id(self, run: mlflow.ActiveRun):
        run_id_path = self.run_dir.joinpath('run_id.txt')
        with open(run_id_path, 'w') as f:
            f.write(run.info.run_id)

    def _get_callbacks(self,
                       model_builder: ModelBuilderBase,
                       active_run: mlflow.ActiveRun):
        """Makes sure that TensorBoard and ModelCheckpoint callbacks exist and are correctly configured.

        Attributes:
            model_builder: ``ModelBuilder`` object, to get callbacks list using ``model_builder.get_callbacks``

        modifies ``callbacks`` to be a list of callbacks, in which ``TensorBoard`` callback exists with
         ``log_dir=self.tensorboard_log_dir`` and ``ModelCheckpoint`` callback exists with
          ``filepath=self.checkpoints_dir/...``, ``save_weights_only=False``

        """

        class MLFlowLogging(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metrics(logs, epoch)

        # class DiscordLogging(tf.keras.callbacks.Callback):
        #     def on_epoch_end(self, epoch, logs=None):
        #         exp_id = active_run.info.experiment_id
        #         exp_name = mlflow.get_experiment(exp_id).name
        #
        #         message = '\n'
        #         for k, v in logs.items():
        #             message += f'  ✌🏼 **{k}** -> {v}\n'
        #         bot.send_message(exp_name=exp_name,
        #                          run_id=active_run.info.run_id,
        #                          epoch=epoch,
        #                          message=message)

        # bot = self.bot

        callbacks = model_builder.get_callbacks()

        mc_callbacks = [i for i in callbacks if isinstance(i, tfkc.ModelCheckpoint)]
        tb_callbacks = [i for i in callbacks if isinstance(i, tfkc.TensorBoard)]

        to_track = self.config.trainer.export_metric
        checkpoint_path = str(self.checkpoints_dir) + "/sm-{epoch:04d}"
        checkpoint_path = checkpoint_path + "-{" + to_track + ":4.5f}"

        if any(mc_callbacks):
            mc_callbacks[0].filepath = str(checkpoint_path)
            mc_callbacks[0].save_weights_only = False
        else:
            mc = tfkc.ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=False)
            callbacks.append(mc)

        if any(tb_callbacks):
            tb_callbacks[0].log_dir = self.tensorboard_log_dir
        else:
            tb = tfkc.TensorBoard(log_dir=self.tensorboard_log_dir)
            callbacks.append(tb)

        mlflow_logging_callback = MLFlowLogging()
        callbacks.append(mlflow_logging_callback)

        # discord_callback = DiscordLogging()
        # callbacks.append(discord_callback)

        return callbacks

    def _load_latest_model(self):
        """Loads and returns the latest ``SavedModel``.

        Returns:
            model: a ``tf.keras.Model`` object.
            initial_epoch: initial epoch for this checkpoint

        """

        latest_ch = self._get_latest_checkpoint()
        initial_epoch = latest_ch['epoch']
        sm_path = latest_ch['path']
        print(f'found latest checkpoint: {sm_path}')
        print(f'resuming from epoch {initial_epoch}')
        model = tfk.models.load_model(latest_ch['path'])
        return model, initial_epoch

    def _get_latest_checkpoint(self):
        """Returns info about the latest checkpoint.

        Returns:
            a dictionary containing epoch, path to ``SavedModel`` and value of ``self.export_metric`` for
            latest checkpoint:
                {'epoch': int, 'path': pathlib.Path, 'value': float}

        """

        checkpoints = get_checkpoints_info(self.checkpoints_dir)
        return max(checkpoints, key=lambda x: os.path.getctime(x['path']))


class Exporter:
    """Exports the best checkpoint as a `mlflow.pyfunc`, """

    def __init__(self, config: DictConfig, run_dir: Path, exported_dir: Path = Path('exported')):
        self.config = config

        self.exported_dir = exported_dir

        self.checkpoints_dir = run_dir.joinpath('checkpoints')
        self.tensorboard_log_dir = run_dir.joinpath('logs')
        self.exported_model_path = exported_dir.joinpath('savedmodel')

    def log_model_to_mlflow(self,
                            active_run: mlflow.ActiveRun,
                            pyfunc_model: mlflow.pyfunc.PythonModel,
                            config_path: Path,
                            signature: typing.Optional[mlflow.models.ModelSignature] = None,
                            mlflow_pyfunc_model_path: str = "tfsm_mlflow_pyfunc") -> mlflow.models.model.ModelInfo:
        """Logs the best model from `self.checkpoints_dir` to the given active_run as an artifact.

        Notes:
            - you can load and use the model by `loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)`
        """

        best_model_info = self.get_best_checkpoint()
        print(f'exporting the {best_model_info["path"]} ..')

        with active_run as _:
            artifacts = {
                "savedmodel_path": str(best_model_info['path']),
                "config_path": str(config_path)
            }

            model_info = mlflow.pyfunc.log_model(
                artifact_path=mlflow_pyfunc_model_path,
                python_model=pyfunc_model,
                artifacts=artifacts,
                signature=signature,
                code_path=['src'])

        return model_info

    def export(self) -> Path:
        """Exports the best version of ``SavedModel`` s, and ``config.yaml`` file into exported sub_directory.

        This method will delete all checkpoints after exporting the best one.
        """

        self.check_for_exported()

        best_model_info = self.get_best_checkpoint()

        # exported_config_path = self.initial_export_dir.joinpath('config.yaml')
        shutil.copytree(best_model_info['path'], self.exported_model_path,
                        symlinks=False, ignore=None, ignore_dangling_symlinks=False)
        # self._write_dict_to_yaml(dict_config, exported_config_path)

        # Delete checkpoints
        shutil.rmtree(self.checkpoints_dir)
        return self.exported_model_path

    def check_for_exported(self):
        """Raises exception if exported directory exists and contains ``savedmodel``"""

        if self.exported_dir.is_dir():
            if any(self.exported_dir.iterdir()):
                if self.exported_model_path.exists():
                    raise Exception('exported savedmodel already exist.')

    def get_best_checkpoint(self):
        """Returns info about the best checkpoint.

        Returns:
            a dictionary containing epoch, path to ``SavedModel`` and value of ``self.export_metric`` for
            the best checkpoint in terms of ``self.export_metric``:
                {'epoch': int, 'path': pathlib.Path, 'value': float}

        """

        checkpoints = get_checkpoints_info(self.checkpoints_dir)

        if self.config.trainer.export_mode == 'min':
            selected_model = min(checkpoints, key=lambda x: x['value'])
        else:
            selected_model = max(checkpoints, key=lambda x: x['value'])
        return selected_model


def get_checkpoints_info(checkpoints_dir: Path):
    """Returns info about checkpoints.

    Returns:
        A list of dictionaries related to each checkpoint:
            {'epoch': int, 'path': pathlib.Path, 'value': float}

    """

    checkpoints = _get_checkpoints(checkpoints_dir)
    ckpt_info = list()
    for cp in checkpoints:
        splits = str(cp.name).split('-')
        epoch = int(splits[1])
        metric_value = float(splits[2])
        ckpt_info.append({'path': cp, 'epoch': epoch, 'value': metric_value})
    return ckpt_info


def _get_checkpoints(checkpoints_dir: Path):
    """Returns a list of paths to folders containing a ``saved_model.pb``"""

    ckpts = [item for item in checkpoints_dir.iterdir() if any(item.glob('saved_model.pb'))]
    return ckpts
