from enum import StrEnum
from functools import partial
from pathlib import Path

import pandas as pd

from qusi.internal.light_curve_collection import LightCurveObservationCollection, \
    create_constant_label_for_path_function
from qusi.internal.light_curve_dataset import LightCurveDataset, \
    default_light_curve_observation_post_injection_transform
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve


class TransitDatasetClassName(StrEnum):
    TRANSIT = 'transit'
    ECLIPSING_BINARY = 'eclipsing_binary'
    OTHER = 'other'


class TransitDatasetTypeName(StrEnum):
    TEST = 'test'
    VALIDATION = 'validation'
    TRAIN = 'train'


class_to_label_dictionary = {
    TransitDatasetClassName.TRANSIT: 1,
    TransitDatasetClassName.ECLIPSING_BINARY: 0,
    TransitDatasetClassName.OTHER: 0,
}


def get_paths_for_class_and_dataset_type(class_: TransitDatasetClassName, type_: TransitDatasetTypeName) -> list[Path]:
    class_csv_path = Path(f'data/transit_evaluation/{type_}_{class_}.csv')
    class_data_frame = pd.read_csv(class_csv_path, index_col=None)
    paths = list(map(Path, class_data_frame['relative_path'].values.tolist()))[:100]
    return paths


def load_times_and_fluxes_from_path(path):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes


def get_test_dataset():
    type_ = TransitDatasetTypeName.TEST
    train_light_curve_dataset = get_type_dataset(type_)
    return train_light_curve_dataset


def get_validation_dataset():
    type_ = TransitDatasetTypeName.VALIDATION
    train_light_curve_dataset = get_type_dataset(type_)
    return train_light_curve_dataset


def get_train_dataset():
    type_ = TransitDatasetTypeName.TRAIN
    train_light_curve_dataset = get_type_dataset(type_)
    return train_light_curve_dataset


def get_type_dataset(type_: TransitDatasetTypeName):
    light_curve_collections = []
    for class_ in TransitDatasetClassName:
        light_curve_collection = LightCurveObservationCollection.new(
            get_paths_function=partial(get_paths_for_class_and_dataset_type, class_, type_),
            load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
            load_label_from_path_function=create_constant_label_for_path_function(class_to_label_dictionary[class_]))
        light_curve_collections.append(light_curve_collection)
    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=light_curve_collections,
        post_injection_transform=partial(default_light_curve_observation_post_injection_transform, length=3500)
    )
    return train_light_curve_dataset
