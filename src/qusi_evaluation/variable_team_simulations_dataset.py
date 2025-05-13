"""
TODO: Warning, this file had been sitting for a while with changes and was committed without confirming it's current
state.
"""

import re
from functools import partial
from pathlib import Path
from random import Random

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from bokeh.io import show

from qusi.internal.finite_standard_light_curve_observation_dataset import FiniteStandardLightCurveObservationDataset
from qusi.internal.light_curve_collection import LightCurveObservationCollection, \
    create_constant_label_for_path_function
from qusi.internal.light_curve_dataset import LightCurveDataset, \
    default_light_curve_observation_post_injection_transform


def get_train_paths_for_class(class_: str) -> list[Path]:
    all_paths = get_paths_for_class(class_)
    splits = np.split(np.array(all_paths), [int(len(all_paths) * 0.1), int(len(all_paths) * 0.2)])
    train_path = splits[2].tolist()
    return train_path


def get_validation_paths_for_class(class_: str) -> list[Path]:
    all_paths = get_paths_for_class(class_)
    splits = np.split(np.array(all_paths), [int(len(all_paths) * 0.1), int(len(all_paths) * 0.2)])
    validation_path = splits[1].tolist()
    return validation_path


def get_test_paths_for_class(class_: str) -> list[Path]:
    all_paths = get_paths_for_class(class_)
    splits = np.split(np.array(all_paths), [int(len(all_paths) * 0.1), int(len(all_paths) * 0.2)])
    test_path = splits[0].tolist()
    return test_path


def get_paths_for_class(class_: str) -> list[Path]:
    light_curves_root_directory = Path('data/variable_team_simulations')
    pattern = re.compile(fr'{class_}_lightcurves(\d+)?')
    paths: list[Path] = []
    for class_directory_path in light_curves_root_directory.iterdir():
        if class_directory_path.is_dir() and pattern.match(class_directory_path.name):
            paths.extend(list(class_directory_path.glob('*.fits')))
    Random(0).shuffle(paths)
    return paths


def load_i_band_times_and_magnitudes_from_light_curve_path(path: Path) -> (npt.NDArray, npt.NDArray):
    hdu_list = fits.open(path)
    for hdu in hdu_list:
        if hdu.name == 'LC_I':
            hjd_times = hdu.data['HJD']
            times = hjd_times - 2457000  # Switch to BTJD.
            magnitudes = hdu.data['mag']
            # Force native byte order.
            times = times.astype(np.float32)
            magnitudes = magnitudes.astype(np.float32)
            return times, magnitudes
    return np.array([-1], dtype=np.float32), np.array([-1], dtype=np.float32)
    raise ValueError(f'No LC_I table found in FITS file {path}')


def get_train_dataset():
    light_curve_collections = []
    class_index = -1
    for class_ in ['CEP', 'DSCT', 'ECL', 'ELL', 'HB', 'LPV', 'RRLYR', 'T2CEP']:
        if len(get_paths_for_class(class_)) < 100:
            continue
        class_index += 1
        light_curve_collection = LightCurveObservationCollection.new(
            get_paths_function=partial(get_train_paths_for_class, class_),
            load_times_and_fluxes_from_path_function=load_i_band_times_and_magnitudes_from_light_curve_path,
            load_label_from_path_function=create_constant_label_for_path_function(class_index))
        light_curve_collections.append(light_curve_collection)
    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=light_curve_collections,
        post_injection_transform=partial(default_light_curve_observation_post_injection_transform, length=7000)
    )
    return train_light_curve_dataset


def get_validation_dataset():
    light_curve_collections = []
    class_index = -1
    for class_ in ['CEP', 'DSCT', 'ECL', 'ELL', 'HB', 'LPV', 'RRLYR', 'T2CEP']:
        if len(get_paths_for_class(class_)) < 100:
            continue
        class_index += 1
        light_curve_collection = LightCurveObservationCollection.new(
            get_paths_function=partial(get_validation_paths_for_class, class_),
            load_times_and_fluxes_from_path_function=load_i_band_times_and_magnitudes_from_light_curve_path,
            load_label_from_path_function=create_constant_label_for_path_function(class_index))
        light_curve_collections.append(light_curve_collection)
    validation_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=light_curve_collections,
        post_injection_transform=partial(default_light_curve_observation_post_injection_transform, length=7000)
    )
    return validation_light_curve_dataset


def get_test_dataset():
    light_curve_collections = []
    class_index = -1
    for class_ in ['CEP', 'DSCT', 'ECL', 'ELL', 'HB', 'LPV', 'RRLYR', 'T2CEP']:
        if len(get_paths_for_class(class_)) < 100:
            continue
        class_index += 1
        light_curve_collection = LightCurveObservationCollection.new(
            get_paths_function=partial(get_test_paths_for_class, class_),
            load_times_and_fluxes_from_path_function=load_i_band_times_and_magnitudes_from_light_curve_path,
            load_label_from_path_function=create_constant_label_for_path_function(class_index))
        light_curve_collections.append(light_curve_collection)
    test_light_curve_dataset = FiniteStandardLightCurveObservationDataset.new(
        light_curve_collections=light_curve_collections,
        post_injection_transform=partial(default_light_curve_observation_post_injection_transform, length=7000)
    )
    return test_light_curve_dataset


if __name__ == '__main__':
    classes = ['CEP', 'DSCT', 'ECL', 'ELL', 'HB', 'LPV', 'RRLYR', 'T2CEP']
    lengths = []
    no_i_band_count = 0
    class_count_dictionary = {}
    for class_ in classes:
        class_count_dictionary[class_] = 0
        class_paths = get_paths_for_class(class_)
        for class_path in class_paths:
            try:
                times, magnitudes = load_i_band_times_and_magnitudes_from_light_curve_path(class_path)
            except ValueError:
                no_i_band_count += 1
                continue
            lengths.append(times.shape[0])
            if len(lengths) % 1000 == 0:
                print(len(lengths))
            class_count_dictionary[class_] += 1
    # histogram_figure = create_histogram_figure(lengths)
    # show(histogram_figure)
    print(len(lengths))
    print(np.median(lengths))
    print(class_count_dictionary)
