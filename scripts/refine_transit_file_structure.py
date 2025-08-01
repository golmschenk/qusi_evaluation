import shutil
from pathlib import Path

import pandas as pd


def main():
    dataset_root_directory = Path('general_light_curve_benchmark_dataset_collection_tess_transiting_planet_dataset')
    dataset_root_directory.mkdir(exist_ok=True)
    dataset_data_directory = dataset_root_directory.joinpath('data')
    dataset_data_directory.mkdir(exist_ok=True)
    light_curve_directory = dataset_data_directory.joinpath('light_curves')
    light_curve_directory.mkdir(exist_ok=True)
    for csv_path in Path('data/transit_evaluation').glob('*.csv'):
        data_frame = pd.read_csv(csv_path, index_col=None)
        data_frame['file_name'] = data_frame['relative_path'].apply(relative_path_string_to_file_name)
        for relative_path_string in data_frame['relative_path']:
            relative_path = Path(relative_path_string)
            shutil.copy(relative_path, light_curve_directory.joinpath(relative_path.name))
        data_frame = data_frame.drop(columns=['relative_path'])
        data_frame.to_csv(dataset_data_directory.joinpath(csv_path.name))


def relative_path_string_to_file_name(relative_path_string: str) -> str:
    return Path(relative_path_string).name


if __name__ == '__main__':
    main()
