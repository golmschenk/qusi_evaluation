"""
TODO: Warning, this file had been sitting for a while with changes and was committed without confirming it's current
state.
"""

from enum import StrEnum
from functools import partial
from pathlib import Path

import pandas as pd
from merida.lightcurves_cls import LightCurvesNExSciLocalFeather

from qusi.internal.light_curve_collection import LightCurveObservationCollection, \
    create_constant_label_for_path_function
from qusi.internal.light_curve_dataset import LightCurveDataset, \
    default_light_curve_observation_post_injection_transform


class MoaMicrolensingDatasetClassName(StrEnum):
    MICROLENSING = 'microlensing'
    HARD_NEGATIVE = 'hard_negative'
    NEGATIVE = 'negative'


class MoaMicrolensingDatasetTypeName(StrEnum):
    TEST = 'test'
    VALIDATION = 'validation'
    TRAIN = 'train'


class_to_label_dictionary = {
    MoaMicrolensingDatasetClassName.MICROLENSING: 1,
    MoaMicrolensingDatasetClassName.HARD_NEGATIVE: 0,
    MoaMicrolensingDatasetClassName.NEGATIVE: 0,
}


def get_paths_for_class_and_dataset_type(class_: MoaMicrolensingDatasetClassName, type_: MoaMicrolensingDatasetTypeName) -> list[Path]:
    class_csv_path = Path(f'data/transit_evaluation/{type_}_{class_}.csv')
    class_data_frame = pd.read_csv(class_csv_path, index_col=None)
    unchecked_paths = list(map(Path, class_data_frame['relative_path'].values.tolist()))
    paths: list[Path] = []
    for unchecked_path in unchecked_paths:
        local_path = Path('data/spoc_sector_27_to_55_light_curves').joinpath(unchecked_path.name)
        if local_path.exists():
            paths.append(local_path)
    return paths


def load_times_and_fluxes_from_path(path):
    light_curve = LightCurvesNExSciLocalFeather(lightcurve_name_=path.stem, lightcurve_class_='', data_path_=path.parent)
    times, fluxes, corrected_fluxes, flux_errors = light_curve.get_days_fluxes_errors()
    return times, corrected_fluxes


def get_test_dataset():
    type_ = MoaMicrolensingDatasetTypeName.TEST
    train_light_curve_dataset = get_type_dataset(type_)
    return train_light_curve_dataset


def get_validation_dataset():
    type_ = MoaMicrolensingDatasetTypeName.VALIDATION
    train_light_curve_dataset = get_type_dataset(type_)
    return train_light_curve_dataset


def get_train_dataset():
    type_ = MoaMicrolensingDatasetTypeName.TRAIN
    train_light_curve_dataset = get_type_dataset(type_)
    return train_light_curve_dataset


def get_train_dataset_with_injection():
    type_ = MoaMicrolensingDatasetTypeName.TRAIN
    light_curve_collections = []
    for class_ in MoaMicrolensingDatasetClassName:
        light_curve_collection = LightCurveObservationCollection.new(
            get_paths_function=partial(get_paths_for_class_and_dataset_type, class_, type_),
            load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
            load_label_from_path_function=create_constant_label_for_path_function(class_to_label_dictionary[class_]))
        light_curve_collections.append(light_curve_collection)
    light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=light_curve_collections,
        post_injection_transform=partial(default_light_curve_observation_post_injection_transform, length=18000)
    )
    return light_curve_dataset


def get_type_dataset(type_: MoaMicrolensingDatasetTypeName):
    light_curve_collections = []
    for class_ in MoaMicrolensingDatasetClassName:
        light_curve_collection = LightCurveObservationCollection.new(
            get_paths_function=partial(get_paths_for_class_and_dataset_type, class_, type_),
            load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
            load_label_from_path_function=create_constant_label_for_path_function(class_to_label_dictionary[class_]))
        light_curve_collections.append(light_curve_collection)
    light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=light_curve_collections,
        post_injection_transform=partial(default_light_curve_observation_post_injection_transform, length=18000)
    )
    return light_curve_dataset
