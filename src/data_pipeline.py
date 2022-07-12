import typing
from pathlib import Path

from omegaconf.dictconfig import DictConfig
import tensorflow as tf
import pandas as pd
import numpy as np
import SimpleITK as sitk
import cv2
import numpy as np
import os
from tensorflow.keras.utils import Sequence
from PIL import Image, ImageOps
from skimage.transform import resize
from imgaug import augmenters as iaa
from os.path import isfile, join
from os import listdir

import numpy as np
import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.metrics import AUC, Recall, Precision, BinaryCrossentropy
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import datetime


def window_img(img, img_min, img_max):
    img = np.clip(img, img_min, img_max)
    return img


sometimes = lambda aug: iaa.Sometimes(0.25, aug)
augmentation = iaa.Sequential([iaa.Fliplr(0.25),
                               iaa.Flipud(0.10),
                               sometimes(iaa.Crop(px=(0, 25), keep_size=True, sample_independently=False))
                               ], random_order=True)


def adj_slice(input_file):
    folder = input_file.split('/')[:-1]
    a = ''
    for strings in folder:
        a += strings + '/'

    center_file = input_file.split('/')[-1]
    onlyfiles = [f for f in listdir(a) if isfile(join(a, f))]
    center_index = onlyfiles.index(center_file)
    re = center_file.split('_')[1].split('.')[0]

    if int(re) == (len(onlyfiles) - 2) or int(re) < 2:
        consecutive_slices = [False]
        labels = [False]
    else:
        two = int(re) - 2
        three = int(re) - 1
        five = int(re) + 1
        six = int(re) + 2

        # consecutive_slices = [(a + 'slice_' + str(two) + '.jpg'), (a + 'slice_' + str(three) + '.jpg'), (a  + onlyfiles[center_index]), (a + 'slice_' + str(five) + '.jpg'), (a + 'slice_' + str(six) + '.jpg')]
        consecutive_slices = [(a + 'slice_' + str(three) + '.jpg'), (a + onlyfiles[center_index]),
                              (a + 'slice_' + str(five) + '.jpg')]
    return consecutive_slices


def read_png(path):
    # slices = adj_slice(path)
    # list_images = []
    # if slices[0] is not False:
    #     for slicee in slices:
    #         image_total = cv2.imread(slicee)
    #         list_images.append(image_total)
    #     a = np.array(list_images)
    #     if len(a.shape) == 4:
    #         return a
    image = cv2.imread(path)
    return image / 255


class AugmentedImageSequence(Sequence):

    def __init__(self, dataset_csv_file, x_names, class_names, source_image_dir, batch_size=16,
                 target_size=(224, 224), augmenter=True, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=2):

        self.dataframe = pd.read_csv(dataset_csv_file)
        self.dataset_df = self.dataframe
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.x_names = x_names
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.load_image(x_path) for x_path in batch_x_path]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        indecies_to_delete = []
        for index, i in enumerate(batch_x):
            if type(i) == np.ndarray:
                pass
            else:
                indecies_to_delete.append(index)

        batch_x = [ele for ele in batch_x if type(ele) == np.ndarray]

        batch_x = np.asarray(batch_x)
        batch_y = np.delete(batch_y, indecies_to_delete, axis=0)

        batch_x = self.transform_batch_images(batch_x)

        return batch_x, batch_y

    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir + image_file)
        image = read_png(image_path)
        return image

    def transform_batch_images(self, batch_x):
        if self.augmenter:
            batch_x = augmentation.augment_images(batch_x)
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps * self.batch_size, :]

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.x_path, self.y = df[self.x_names].values, df[self.class_names].values.astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()


class StepCalculator:

    @staticmethod
    def get_sample_counts(dataset, class_names):
        df = dataset
        total_count = df.shape[0]
        labels = df[class_names].values
        positive_counts = np.sum(labels, axis=0)
        class_positive_counts = dict(zip(class_names, positive_counts))
        return total_count, class_positive_counts

    @staticmethod
    def calculating_class_weights(y_true):
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            weights[i] = compute_class_weight('balanced', classes=np.unique(y_true[:, i]), y=y_true[:, i])
        return weights

    @staticmethod
    def get_weighted_loss(weights):
        def weighted_loss(y_true, y_pred):
            return K.mean(
                (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
                axis=-1)

        return weighted_loss

    @staticmethod
    def metrics_define(num_classes):
        metrics_all = ['accuracy',
                       AUC(curve='PR', multi_label=True, num_labels=num_classes, name='auc_pr'),
                       AUC(multi_label=True, num_labels=num_classes, name='auc_roc'),
                       ]
        return metrics_all

    def calculate_steps(self, config):
        train_df = pd.read_csv(config.data_pipeline.train_csv)
        validation_df = pd.read_csv(config.data_pipeline.validation_csv)
        train_counts, train_pos_counts = self.get_sample_counts(train_df, config.class_names)
        dev_counts, dev_pos_counts = self.get_sample_counts(validation_df, config.class_names)
        train_steps = int(np.ceil(train_counts / config.data_pipeline.train_batch_size))
        validation_steps = int(np.ceil(dev_counts / config.data_pipeline.val_batch_size))
        weights = self.calculating_class_weights(train_df[config.class_names].values.astype(np.float32))
        weights = np.sqrt(weights)
        return train_steps, validation_steps, weights


class DataLoader:
    def __init__(self, config):
        self.config = config

    def create_data(self):
        step_calculator = StepCalculator()
        n_tr, n_val, weights = step_calculator.calculate_steps(self.config)

        tr_gen = AugmentedImageSequence(
            dataset_csv_file=self.config.data_pipeline.train_csv,
            x_names='imgfile',
            class_names=self.config.class_names,
            source_image_dir='',
            batch_size=self.config.data_pipeline.train_batch_size,
            target_size=(self.config.data_pipeline.image_dimension, self.config.data_pipeline.image_dimension),
            augmenter=False,
            shuffle_on_epoch_end=True,
            steps=n_tr,
        )

        val_gen = AugmentedImageSequence(
            dataset_csv_file=self.config.data_pipeline.validation_csv,
            x_names='imgfile',
            class_names=self.config.class_names,
            source_image_dir='',
            batch_size=self.config.data_pipeline.val_batch_size,
            target_size=(self.config.data_pipeline.image_dimension, self.config.data_pipeline.image_dimension),
            steps=n_val,
            shuffle_on_epoch_end=True,
            augmenter=False,
        )
        return tr_gen, val_gen, n_tr, n_val

