import math
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from qusi_evaluation.transit_dataset import load_times_and_fluxes_from_path as transit_load_times_and_fluxes_from_path


def show_light_curve_statistics(
        *,
        dataset_root_directory: Path,
        light_curve_extension: str = 'fits',
        load_light_curve_function: Callable[[Path], tuple[npt.NDArray, npt.NDArray]],
):
    lengths = []
    nan_time_exists = False
    nan_flux_exists = False
    maximum_flux = -math.inf
    minimum_flux = math.inf
    maximum_time = -math.inf
    minimum_time = math.inf
    for light_curve_path in dataset_root_directory.glob(f'**/*.{light_curve_extension}'):
        times, fluxes = load_light_curve_function(light_curve_path)
        lengths.append(times.shape[0])
        if np.max(fluxes) > maximum_flux:
            maximum_flux = np.nanmax(fluxes)
        if np.min(fluxes) < minimum_flux:
            minimum_flux = np.nanmin(fluxes)
        if np.max(times) > maximum_time:
            maximum_time = np.nanmax(times)
        if np.min(times) < minimum_time:
            minimum_time = np.nanmin(times)
        if not nan_time_exists:
            if np.any(np.isnan(times)):
                nan_time_exists = True
        if not nan_flux_exists:
            if np.any(np.isnan(fluxes)):
                nan_flux_exists = True
    print(f'Light curve count: {len(lengths)}')
    print(f'Light curve mean length: {np.mean(lengths)}')
    print(f'Light curve maximum length: {np.max(lengths)}')
    print(f'Light curve minimum length: {np.min(lengths)}')
    print(f'Light curve maximum time: {maximum_time}')
    print(f'Light curve minimum time: {minimum_time}')
    print(f'Light curve NaN time exists: {nan_time_exists}')
    print(f'Light curve maximum flux: {maximum_flux}')
    print(f'Light curve minimum flux: {minimum_flux}')
    print(f'Light curve NaN flux exists: {nan_flux_exists}')



if __name__ == '__main__':
    show_light_curve_statistics(
        dataset_root_directory=Path('data/general_light_curve_benchmark_dataset_collection_tess_transit_dataset'),
        light_curve_extension='fits',
        load_light_curve_function=transit_load_times_and_fluxes_from_path,
    )
