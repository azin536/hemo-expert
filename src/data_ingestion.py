from pathlib import Path
import os
import typing

import numpy as np
import SimpleITK as sitk
import pandas as pd
from pydicom import dcmread, Dataset
import cv2

from . import DATA_FOLDER

image_type_ = 'AXIAL'
modality_ = 'CT'
body_part_ = 'HEAD'


def prepare_data(meta_data_file: pd.DataFrame,
                 output_dir: Path):
    """Prepares data to be read via the DataLoader.

    this function works on a meta_data file which has label information, and generates the train/eval sub-folders on the
    output_dir.
    """

    preprocessor = Preprocessor()

    train_out_dir = Path(output_dir).joinpath('train')
    train_out_dir.mkdir(exist_ok=True, parents=True)
    eval_out_dir = Path(output_dir).joinpath('eval')
    eval_out_dir.mkdir(exist_ok=True, parents=True)

    unique_series = meta_data_file['SeriesInstanceUID'].unique().tolist()
    path_label = list()

    for sid in unique_series:
        print(f'preparing {sid} series ...')
        sub_df = meta_data_file[meta_data_file['SeriesInstanceUID'] == sid]
        data_source = sub_df['DataSource'].values.tolist()[0]
        split = sub_df['Split'].values.tolist()[0]

        series_path = DATA_FOLDER.joinpath(data_source).joinpath(sid)
        try:
            validate_rg_input(series_path)
        except Exception as e:
            print(f'input validation exception for series {sid}: {e.args[0]}')
        else:
            preprocessed = preprocessor.prepare_series(series_path)
            for ind, slice in enumerate(preprocessed):
                file_name = f'{sid}_{ind}.png'
                if 'train' in split.lower():
                    output_path = train_out_dir.joinpath(file_name)
                else:
                    output_path = eval_out_dir.joinpath(file_name)
                # output_path = Path(output_dir).joinpath(split).joinpath(file_name)
                cv2.imwrite(str(output_path), slice * 255)

                label = sub_df[sub_df['SliceIndex'] == ind]['Label'].values[0]
                path_label.append({'Path': str(Path(split).joinpath(file_name)),
                                   'Label': label.lower(),
                                   'Split': split.lower()})

    pd.DataFrame(path_label).to_csv(train_out_dir.parent.joinpath('labels.csv'), index=False)


class Preprocessor:
    """Turning a folder of DICOM series into numpy array of raw values, and a header file."""

    def __init__(self):
        self.brain_window = (0, 80)
        self.subdural_window = (-20, 180)
        self.soft_window = (-150, 230)
        self.target_size = 256

    def prepare_series(self, series_path: Path) -> np.ndarray:
        """

        Args:
            series_path: numpy series with shape(n_slices, 512, 512), in hounsfield units

        Returns:
            numpy array of shape(n_slices, target_size, target_size, 3), in range(0, 1)

        """

        numpy_series, header = self.get_series_as_array(series_path)

        down_sampled = self.downsample_series(numpy_series)

        brain_volume = np.clip(down_sampled, self.brain_window[0], self.brain_window[1])
        brain_volume = (brain_volume - self.brain_window[0]) / (self.brain_window[1] - self.brain_window[0])

        subdural_volume = np.clip(down_sampled, self.subdural_window[0], self.subdural_window[1])
        subdural_volume = (subdural_volume - self.subdural_window[0]) / (
                self.subdural_window[1] - self.subdural_window[0])

        soft_volume = np.clip(down_sampled, self.soft_window[0], self.soft_window[1])
        soft_volume = (soft_volume - self.soft_window[0]) / (self.soft_window[1] - self.soft_window[0])

        final_volume = np.stack([brain_volume, subdural_volume, soft_volume], axis=-1)
        return final_volume

    def downsample_series(self, series: np.ndarray) -> np.ndarray:
        """Downsamples the input series to the (self.target_size, self.target_size)"""

        size = (self.target_size, self.target_size)

        series = np.moveaxis(series, 0, 2)
        series = sitk.GetImageFromArray(series, sitk.sitkInt32)

        original_ct = series
        dimension = original_ct.GetDimension()
        reference_physical_size = np.zeros(original_ct.GetDimension())
        reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                      zip(original_ct.GetSize(), original_ct.GetSpacing(),
                                          reference_physical_size)]
        reference_origin = original_ct.GetOrigin()
        reference_direction = original_ct.GetDirection()
        reference_size = size
        reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]
        reference_image = sitk.Image(reference_size, original_ct.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)
        reference_center = np.array(
            reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(original_ct.GetDirection())
        transform.SetTranslation(np.array(original_ct.GetOrigin()) - reference_origin)
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(
            original_ct.TransformContinuousIndexToPhysicalPoint(np.array(original_ct.GetSize()) / 2.0))
        centering_transform.SetOffset(
            np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.CompositeTransform([transform, centering_transform])
        resampled_img = sitk.Resample(original_ct, reference_image, centered_transform, sitk.sitkLinear, 0.0)

        series_arr = np.moveaxis(sitk.GetArrayFromImage(resampled_img), 2, 0)
        return series_arr

    def get_series_as_array(self, series_path: Path) -> typing.Tuple[np.ndarray, Dataset]:
        """This method applies the base preprocessing to the series.

        Notes:
            - this method makes this assumption: only one dicom series exists in the given path.
            - this method returns an array of shape(n_slices, 512, 512) with hounsfield units
            - the header will be read from the first DICOM file of the series
            - no transformations, just reading and returning as numpy

        """

        dicoms = list(series_path.glob('*.dcm'))
        header = dcmread(str(dicoms[0]), stop_before_pixels=True)

        series = self.read_series(series_path)
        arr = np.moveaxis(sitk.GetArrayFromImage(series), 2, 0)
        return arr, header

    @staticmethod
    def read_series(series_path: Path) -> sitk.Image:
        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(series_path))[0]
        reader = sitk.ImageSeriesReader()
        image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(str(series_path), series_id), sitk.sitkInt32)
        image = np.array(sitk.GetArrayFromImage(image), dtype=np.float32)
        image = np.moveaxis(image, 0, 2)
        image = sitk.GetImageFromArray(image, sitk.sitkInt32)
        return image


class InputValidationError(Exception):
    def __init__(self, int_msg, ext_msg):
        super().__init__(ext_msg, int_msg)


def validate_rg_input(series_path: Path,
                      dcm_tags_to_check: dict = {'Modality': 'CT'},
                      min_dcm_files_for_series: int = 15) -> str:
    """Makes sure that there is only one DICOM series in the ``series_path`` by removing the extra series.

    Notes: - the remained DICOM series is guaranteed to have the same Modality, BodyPartExamined, and ImageType as
    determined in the package's ``__init__.py`` file. - use this before feeding the series into the Preprocessor,
    just after unzipping
    """

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(series_path))
    print(f'found {len(series_ids)} series.')

    for sid in series_ids:
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_path), sid)
        dcm_file = dcmread(series_file_names[0], stop_before_pixels=True)

        conditions = list()
        for tag in dcm_tags_to_check:
            conditions.append(eval('dcm_file.' + tag + '.upper()') == dcm_tags_to_check[tag].upper())
        conditions.append(image_type_.upper() in [i.upper() for i in dcm_file.ImageType])

        if all(conditions):
            print(f'series {sid} is safe')
        else:
            print(f'removing the {sid} series, not all conditions passed')
            for file_path in series_file_names:
                os.remove(file_path)

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(series_path))
    if len(series_ids) > 1:
        raise InputValidationError(int_msg=f'two {image_type_}-{modality_}-{body_part_} series',
                                   ext_msg=f'There is two {image_type_}-{modality_}-{body_part_} series')

    elif not any(series_ids):
        raise InputValidationError(int_msg=f'no {image_type_}-{modality_}-{body_part_} series in the extracted file.',
                                   ext_msg=f'Could not find any {image_type_}-{modality_}-{body_part_} series')

    else:
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_path), series_ids[0])
        n_dcm_files = len(series_file_names)
        if n_dcm_files < min_dcm_files_for_series:
            raise InputValidationError(int_msg=f'less than {n_dcm_files} DICOM files in the series',
                                       ext_msg=f'Not enough slices for the series: {n_dcm_files}')
        else:
            sample_dcm_file = dcmread(series_file_names[0], stop_before_pixels=True)
            try:
                study_description = sample_dcm_file.StudyDescription
            except AttributeError:
                print('could not load the StudyDescription field')
            else:
                print(f'StudyDescription: {study_description}')
                return series_ids[0]
