import math
import numpy as np
import numpy.typing as npt
from bokeh.io import save
from bokeh.models import Column, Row, Div
from pathlib import Path
from typing import Callable

from gobo.high_level import histogram


def show_light_curve_statistics(
        *,
        dataset_root_directory: Path,
        light_curve_extension: str = 'fits',
        load_light_curve_function: Callable[[Path], tuple[npt.NDArray, npt.NDArray]],
) -> None:
    (lengths, non_nan_flux_lengths, maximum_flux, maximum_time, minimum_flux, minimum_time, nan_flux_exists,
     nan_time_exists
     ) = get_light_curve_statistics(dataset_root_directory, light_curve_extension, load_light_curve_function)
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


def create_light_curve_statistics_report(
        *,
        dataset_root_directory: Path,
        light_curve_extension: str = 'fits',
        load_light_curve_function: Callable[[Path], tuple[npt.NDArray, npt.NDArray]],
) -> None:
    (lengths, non_nan_flux_lengths, maximum_flux, maximum_time, minimum_flux, minimum_time, nan_flux_exists,
     nan_time_exists
     ) = get_light_curve_statistics(dataset_root_directory, light_curve_extension, load_light_curve_function)
    lengths = np.array(lengths)
    lengths_histogram_figure = histogram(lengths, title='Lengths', x_axis_label='Count', y_axis_label='Length')
    non_nan_flux_lengths = np.array(non_nan_flux_lengths)
    non_nan_flux_lengths_histogram_figure = histogram(non_nan_flux_lengths, title='Non-NaN flux lengths', x_axis_label='Count', y_axis_label='Length')
    table_dictionary = {
        'Light curve count': len(lengths),
        'Light curve mean length': np.mean(lengths),
        'Light curve maximum length': np.max(lengths),
        'Light curve minimum length': np.min(lengths),
        'Light curve maximum time': maximum_time,
        'Light curve minimum time': minimum_time,
        'Light curve NaN time exists': nan_time_exists,
        'Light curve maximum flux': maximum_flux,
        'Light curve minimum flux': minimum_flux,
        'Light curve NaN flux exists': nan_flux_exists
    }
    div_text = '<table>'
    for name, value in table_dictionary.items():
        div_text += f'<tr><td>{name}:</td><td>{value}</td></tr>'
    div_text += '</table>'
    main_statistics_table = Div(text=div_text, styles={'font-size': '140%'})
    row = Row(lengths_histogram_figure, non_nan_flux_lengths_histogram_figure)
    column = Column(main_statistics_table, row)
    save(column, f'{dataset_root_directory.name}.html')


def get_light_curve_statistics(
        dataset_root_directory: Path, light_curve_extension: str,
        load_light_curve_function: Callable[[Path], tuple[npt.NDArray, npt.NDArray]]
) -> tuple[list[int], list[int], float, float, bool, float, float, bool]:
    lengths = []
    non_nan_flux_lengths = []
    nan_time_exists = False
    nan_flux_exists = False
    maximum_flux = -math.inf
    minimum_flux = math.inf
    maximum_time = -math.inf
    minimum_time = math.inf
    for light_curve_path in dataset_root_directory.glob(f'**/*.{light_curve_extension}'):
        print(light_curve_path)
        times, fluxes = load_light_curve_function(light_curve_path)
        lengths.append(times.shape[0])
        non_nan_flux_lengths.append(fluxes[~np.isnan(fluxes)].shape[0])  # TODO: Add plots of distributions
        if np.nanmax(fluxes) > maximum_flux:
            maximum_flux = np.nanmax(fluxes)
        if np.nanmin(fluxes) < minimum_flux:
            minimum_flux = np.nanmin(fluxes)
        if np.nanmax(times) > maximum_time:
            maximum_time = np.nanmax(times)
        if np.nanmin(times) < minimum_time:
            minimum_time = np.nanmin(times)
        if not nan_time_exists:
            if np.any(np.isnan(times)):
                nan_time_exists = True
        if not nan_flux_exists:
            if np.any(np.isnan(fluxes)):
                nan_flux_exists = True
    return lengths, non_nan_flux_lengths, maximum_flux, maximum_time, minimum_flux, minimum_time, nan_flux_exists, nan_time_exists
