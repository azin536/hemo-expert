from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
import pandas as pd

import os
import SimpleITK as sitk
import numpy
import numpy as np
from scipy import ndimage
from skimage import morphology
import cv2


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
    series_path = '/'.join(folder)

    images_IDs = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_path)
    center_index = images_IDs.index(input_file)

    if center_index >= (len(images_IDs) - 2) or center_index < 2:
        consecutive_slices = [False]
    else:
        one = center_index - 1
        three = center_index + 1

        consecutive_slices = [images_IDs[one], images_IDs[center_index], images_IDs[three]]
    return consecutive_slices


def read_png(path):
    slices = adj_slice(path)
    list_images = []
    if slices[0] is not False:
        for slicee in slices:
            image = sitk.ReadImage(str(slicee))
            image = sitk.GetArrayFromImage(image)[0]
            brain_image = np.clip(image, 0, 80)
            segmentation = morphology.dilation(brain_image, np.ones((7, 7)))
            segmentation = ndimage.morphology.binary_fill_holes(segmentation)

            labels, label_nb = ndimage.label(segmentation)

            label_count = np.bincount(labels.ravel().astype(np.uint8))
            label_count[0] = 0

            mask = labels == label_count.argmax()
            mask = morphology.dilation(mask, np.ones((1, 1)))
            mask = ndimage.morphology.binary_fill_holes(mask)
            mask = morphology.dilation(mask, np.ones((3, 3)))
            mask = np.uint8(mask)
            a = np.count_nonzero(mask)
            if a == 262144:
                pass
            else:
                contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(contours, key=cv2.contourArea)
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                x, y, w, h = cv2.boundingRect(mask)
                if (40 < angle < 135) and (w > 1.4 * h):
                    pass
                else:
                    ROI = image[y:y + h, x:x + w]
                    pic_back = ROI.min()
                    shifted = np.ones(image.shape) * pic_back
                    x = 256 - ROI.shape[1] // 2
                    y = 256 - ROI.shape[0] // 2
                    shifted[y:y + h, x:x + w] = ROI

                    image_brain = window_img(shifted, 0, 80)
                    image_brain = (image_brain - 0) / 80
                    image_brain = np.expand_dims(image_brain, axis=2)

                    image_subdural = window_img(shifted, -20, 180)
                    image_subdural = (image_subdural - (-20)) / 200
                    image_subdural = np.expand_dims(image_subdural, axis=2)

                    image_soft = window_img(shifted, -150, 230)
                    image_soft = (image_soft - (-150)) / 380
                    image_soft = np.expand_dims(image_soft, axis=2)

                    image_total = np.concatenate([image_brain, image_subdural, image_soft], axis=2)
                    image_total = (image_total * 255).astype(np.uint8)
                    image_total = cv2.resize(image_total, dsize=(256, 256))
                    list_images.append(image_total / 255)
        a = np.array(list_images)
        if len(a.shape) == 4:
            return a


class AugmentedImageSequence(Sequence):

    def __init__(self, dataset_csv_file, x_names, class_names, source_image_dir, batch_size=16,
                 target_size=(224, 224), augmenter=True, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=2):

        self.dataset_df = pd.read_pickle(dataset_csv_file)
        self.dataset_df = self.dataset_df[:40]
        new = []
        root = 'C:/Users/Azin/PycharmProjects/one_class/hemo-expert/data/RSNA_ICH_Dataset/'
        for index, slice_path in enumerate(self.dataset_df['SliceName'].values):
            new.append(root + self.dataset_df['SeriesInstanceUID'].values[index] + '/' + slice_path)
        self.dataset_df['imgfile'] = new

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

    def get_x_true(self):
        """
        Use this function to get x_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.x_path[:self.steps * self.batch_size]

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

    def calculate_steps(self, config):
        train_df = pd.read_pickle(config.data_pipeline.train_csv)
        validation_df = pd.read_pickle(config.data_pipeline.validation_csv)
        evaluation_df = pd.read_pickle(config.evaluation.evaluation_csv)

        train_counts, train_pos_counts = self.get_sample_counts(train_df, config.class_names)
        val_counts, val_pos_counts = self.get_sample_counts(validation_df, config.class_names)
        eval_counts, eval_pos_counts = self.get_sample_counts(evaluation_df, config.class_names)

        train_steps = int(np.ceil(train_counts / config.data_pipeline.train_batch_size))
        validation_steps = int(np.ceil(val_counts / config.data_pipeline.val_batch_size))
        evaluation_steps = int(np.ceil(eval_counts / config.evaluation.eval_batch_size))

        return train_steps, validation_steps, evaluation_steps


class DataLoader:
    def __init__(self, config):
        self.config = config
        step_calculator = StepCalculator()
        self.n_tr, self.n_val, self.n_eval = step_calculator.calculate_steps(self.config)

    def create_data(self):
        tr_gen = AugmentedImageSequence(
            dataset_csv_file=self.config.data_pipeline.train_csv,
            x_names='imgfile',
            class_names=self.config.class_names,
            source_image_dir='',
            batch_size=self.config.data_pipeline.train_batch_size,
            target_size=(self.config.data_pipeline.image_dimension, self.config.data_pipeline.image_dimension),
            augmenter=False,
            shuffle_on_epoch_end=True,
            steps=self.n_tr,
        )

        val_gen = AugmentedImageSequence(
            dataset_csv_file=self.config.data_pipeline.validation_csv,
            x_names='imgfile',
            class_names=self.config.class_names,
            source_image_dir='',
            batch_size=self.config.data_pipeline.val_batch_size,
            target_size=(self.config.data_pipeline.image_dimension, self.config.data_pipeline.image_dimension),
            steps=self.n_val,
            shuffle_on_epoch_end=True,
            augmenter=False,
        )
        return tr_gen, val_gen, self.n_tr, self.n_val

    def create_eval_data(self):
        eval_gen = AugmentedImageSequence(
            dataset_csv_file=self.config.evaluation.evaluation_csv,
            x_names='imgfile',
            class_names=self.config.class_names,
            source_image_dir='',
            batch_size=self.config.evaluation.eval_batch_size,
            target_size=(self.config.data_pipeline.image_dimension, self.config.data_pipeline.image_dimension),
            steps=self.n_eval,
            shuffle_on_epoch_end=False,
            augmenter=False,
        )
        return eval_gen, self.n_eval
