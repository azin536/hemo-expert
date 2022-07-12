from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Callable

from omegaconf.dictconfig import DictConfig
import pandas as pd
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers


def prepare_data(meta_data_file: pd.DataFrame,
                 output_dir: Path,
                 **kwargs):
    """Prepares data to be read via the DataLoader.

    this function works on a meta_data file which has label information, and generates the train/eval sub-folders on the
    output_dir.
    """

    pass


class ModelBuilderBase(ABC):

    """Building and compiling ``tensorflow.keras`` models to train with Trainer.

        Notes:
            - you have to override these methods: ``get_model``
            - you may override these methods too (optional): ``get_callbacks``, ``get_class_weight``
            - don't override the private ``_{method_name}`` methods

        Examples:
            >>> model_builder = ModelBuilderBase(config)
            >>> model = model_builder.get_model()
            >>> callbacks = model_builder.get_callbacks()
            >>> model.fit(train_gen, n_iter_train, callbacks=callbacks, class_weight=class_weight)

        """
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def get_compiled_model(self) -> tfk.Model:
        """Generates the model for training, and returns the compiled model.

        Returns:
            A compiled ``tensorflow.keras`` model.
        """

    def get_callbacks(self) -> list:
        """Returns any callbacks for ``fit``.

        Returns:
            list of ``tf.keras.Callback`` objects. ``Orchestrator`` will handle the ``ModelCheckpoint`` and ``Tensorboard`` callbacks.
            Still, you can return each of these two callbacks, and orchestrator will modify your callbacks if needed.

        """
        return list()

    def get_class_weight(self) -> Optional[dict]:
        """Set this if you want to pass ``class_weight`` to ``fit``.

        Returns:
           Optional dictionary mapping class indices (integers) to a weight (float) value.
           used for weighting the loss function (during training only).
           This can be useful to tell the model to "pay more attention" to samples from an under-represented class.

        """

        return None


class DataLoaderBase(ABC):
    """Data-loading mechanism.

    This class will create data generators.

    Notes:
        - Output of train/validation data generators will be a tuple of (image/volume, label/segmentation_map, sample_weight).
         If you don't need ``sample_weight``, set it to ``1`` for all data-points.
        - Output of evaluation data generator will be a tuple of (image/volume, label/segmentation_map, data_id).
         Each ``data_id`` could be anything specific that can help to retrieve this data point. You can consider to set
          ``data_id=row_id`` of the meta-data's dataframe, if you have one.
        - You can use the third argument with "weighted metrics" in the compile method of the model,
         or for weighted custom loss functions.

    Attributes:
        config (ConfigStruct): config file


    Examples:
        >>> data_loader = DataLoader(config, data_dir)
        >>> train_data_gen, n_iter_train, val_data_gen, n_iter_val = data_loader.create_train_val_generators()
        >>> eval_data_gen, n_iter_eval = data_loader.create_eval_generator()

    """

    def __init__(self, config: DictConfig, data_dir: Path):
        self.config_ = config
        self.config = self.config_.data_pipeline
        self.data_dir = data_dir

    @abstractmethod
    def create_train_val_generators(self) -> Tuple[Union[tf.data.Dataset, tfk.utils.Sequence], int, Union[tf.data.Dataset, tfk.utils.Sequence], int]:
        """Create data generator for training and validation sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
        (x_batch, y_batch, sample_weight(s)_batch)

        Notes:
            - If you don't need ``sample_weight``, set it to ``1`` for all data-points.

        Returns:
            tuple(train_gen, n_iter_train, val_gen, n_iter_val):
            - train_gen: a ``generator``/``tf.data.Dataset``.
            - n_iter_train: number of iterations to complete an epoch.
            - val_gen: a ``generator``/``tf.data.Dataset``.
            - n_iter_val: number of iterations to complete an epoch.

        """

    @abstractmethod
    def create_eval_generator(self):
        """Create data generator for evaluation sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
        (x_batch, y_batch, data_id(str)_batch),

        Notes:
            - Each ``data_id`` could be anything specific that can help to retrieve this data point.
            - You can consider to set ``data_id=row_id`` of the test subset's dataframe, if you have one.
            - Do not repeat this dataset, i.e. raise an exception at the end.

        Returns:
            tuple(generator, n_iter):
            - generator: a ``generator``/``tf.data.Dataset``.
            - n_iter: number of iterations to complete an epoch.

        """


class Metric:

    """Rule: function's input and output must be ``np.ndarray``"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass
        # return self.__doc__

    @abstractmethod
    def get_func(self) -> Callable:
        pass

