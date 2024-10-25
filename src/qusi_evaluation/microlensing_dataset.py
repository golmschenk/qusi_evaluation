from pathlib import Path
import re

import numpy as np
import numpy.typing as npt

from qusi.data import LightCurveDataset, LightCurveObservationCollection
from qusi.experimental.application.tess import TessMissionLightCurve
from qusi_evaluation.download_spoc_sector_27_to_56_light_curve_data import spoc_sector_27_to_55_light_curve_directory


def get_spoc_10_minute_ffi_light_curve_paths():
    paths = list(spoc_sector_27_to_55_light_curve_directory.glob('**/*.fits'))
    return paths


def get_rich_barry_synthetic_microlensing_paths():
    paths: list[Path] = []
    for path in Path('data/microlensing_training_lightcurves').glob('**/*.txt'):
        if re.match(r'Lc_\d+\.txt', path.name):
            paths.append(path)
    print(f'RBMLC: {len(paths)}, flush=True')
    return paths


def load_times_and_fluxes_from_tess_path(path):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes


def load_times_and_magnifications_from_rich_barry_path(path: Path) -> (npt.NDArray, npt.NDArray):
    light_curve = np.loadtxt(path, dtype=np.float32)
    times = light_curve[:, 0]
    fluxes = light_curve[:, 1]
    return times, fluxes


def positive_label_function(path):
    return 1


def negative_label_function(path):
    return 0


def get_microlensing_train_dataset():
    tess_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_spoc_10_minute_ffi_light_curve_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_tess_path,
        load_label_from_path_function=negative_label_function)
    synthetic_microlensing_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_rich_barry_synthetic_microlensing_paths,
        load_times_and_fluxes_from_path_function=load_times_and_magnifications_from_rich_barry_path,
        load_label_from_path_function=positive_label_function)
    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[tess_light_curve_collection],
        injectee_light_curve_collections=[tess_light_curve_collection],
        injectable_light_curve_collections=[synthetic_microlensing_light_curve_collection],
    )
    return train_light_curve_dataset
