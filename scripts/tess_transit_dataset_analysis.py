from pathlib import Path

from qusi_evaluation.internal.dataset_analysis import create_light_curve_statistics_report
from qusi_evaluation.transit_dataset import load_times_and_fluxes_from_path

if __name__ == '__main__':
    create_light_curve_statistics_report(
        dataset_root_directory=Path('data/general_light_curve_benchmark_dataset_collection_tess_transit_dataset'),
        light_curve_extension='fits',
        load_light_curve_function=load_times_and_fluxes_from_path,
    )
