from pathlib import Path

import numpy as np
from qusi.data import FiniteStandardLightCurveDataset, LightCurveDataset, LightCurveCollection
from qusi.experimental.application.tess import TessMissionLightCurve
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import Column


def get_tess_light_curves_paths():
    path_list: list[Path] = []
    glob = Path('data/spoc_sector_27_to_55_light_curves').glob('**/*.fits')
    for path in glob:
        path_list.append(path)
        if len(path_list) > 1_000_000:
            break
    return list(glob)


def get_normal_distribution_paths():
    return list(Path('data/microlensing_pspl_and_normal_light_curves').glob('Gauss_*.txt'))


def get_pspl_paths():
    return list(Path('data/microlensing_pspl_and_normal_light_curves').glob('PSPL_*.txt'))


def get_infer_paths():
    return get_tess_light_curves()


def load_times_and_fluxes_from_path(path):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes


def load_times_and_magnificiations_from_rich_path(path):
    light_curve = np.loadtxt(path)
    return light_curve[:, 0], light_curve[:, 1] + 1


def positive_label_function(path):
    return 1


def negative_label_function(path):
    return 0


def get_rich_train_dataset():
    tess_light_curve_collection = LightCurveObservationCollection.new()
    pspl_light_curve_collection = LightCurveObservationCollection.new()
    normal_light_curve_collection = LightCurveObservationCollection.new()
    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[tess_light_curve_collection],
        injectee_light_curve_collections=[tess_light_curve_collection],
        injectable_light_curve_collections=[pspl_light_curve_collection, normal_light_curve_collection]
    )
    return train_light_curve_dataset


def get_transit_validation_dataset():
    positive_validation_light_curve_collection = LightCurveObservationCollection.new()
    negative_validation_light_curve_collection = LightCurveObservationCollection.new()
    validation_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_validation_light_curve_collection,
                                          negative_validation_light_curve_collection])
    return validation_light_curve_dataset


def get_rich_infer_dataset():
    tess_light_curve_collection = LightCurveCollection.new(
        get_paths_function=get_tess_light_curves_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path)
    test_light_curve_dataset = FiniteStandardLightCurveDataset.new(
        light_curve_collections=[tess_light_curve_collection])
    return test_light_curve_dataset


if __name__ == '__main__':
    figures = []
    for index in range(1, 100):
        pspl_times, pspl_magnifications = load_times_and_magnificiations_from_rich_path(f'data/microlensing_pspl_and_normal_light_curves/PSPL_{index}.txt')
        normal_times, normal_magnifications = load_times_and_magnificiations_from_rich_path(f'data/microlensing_pspl_and_normal_light_curves/Gauss_{index}.txt')
        light_curve_figure = figure(x_axis_label='Time (BTJD)', y_axis_label='Flux')
        light_curve_figure.scatter(x=pspl_times, y=pspl_magnifications, color='mediumblue')
        light_curve_figure.line(x=pspl_times, y=pspl_magnifications, line_alpha=0.3, color='mediumblue')
        light_curve_figure.scatter(x=normal_times, y=normal_magnifications, color='firebrick')
        light_curve_figure.line(x=normal_times, y=normal_magnifications, line_alpha=0.3, color='firebrick')
        figures.append(light_curve_figure)
    column = Column(*figures)
    show(column)
    pass