import itertools
import shutil
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
import requests
from astropy.io import ascii

from ramjet.data_interface.tess_data_interface import get_spoc_tic_id_list_from_mast, \
    download_spoc_light_curves_for_tic_ids
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface, ToiColumns
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve

temporary_data_directory = Path('data/temporary')
dataset_root_path = Path('data/general_light_curve_benchmark_dataset_collection_tess_transit_dataset')
dataset_light_curve_directory = dataset_root_path.joinpath('light_curves')


def get_non_transit_candidate_tic_ids():
    tess_toi_data_interface = TessToiDataInterface()
    non_negative_tic_ids = tess_toi_data_interface.toi_dispositions[
        tess_toi_data_interface.toi_dispositions[ToiColumns.disposition.value] != 'FP'][ToiColumns.tic_id.value]
    spoc_target_tic_ids = get_spoc_tic_id_list_from_mast()
    negative_tic_ids = list(set(spoc_target_tic_ids) - set(non_negative_tic_ids))
    return negative_tic_ids


def get_transit_tic_ids():
    tess_toi_data_interface = TessToiDataInterface()
    positive_tic_ids = tess_toi_data_interface.toi_dispositions[
        (tess_toi_data_interface.toi_dispositions[ToiColumns.disposition.value].isin(['CP', 'KP'])) &
        # Less than a TESS sector.
        (tess_toi_data_interface.toi_dispositions[ToiColumns.transit_period__days.value] <= 27) &
        # Greater than a small number to remove bad data points.
        (tess_toi_data_interface.toi_dispositions[ToiColumns.transit_period__days.value] >= 0.001)
        ][ToiColumns.tic_id.value]
    return np.unique(positive_tic_ids.values).tolist()


def get_eclipsing_binary_tic_ids() -> list[int]:
    kostov_new_eclipsing_binary_table_response = requests.get(
        'https://content.cld.iop.org/journals/0067-0049/279/2/50/revision1/apjsade2d8t3_mrt.txt')
    kostov_known_eclipsing_binary_table_response = requests.get(
        'https://content.cld.iop.org/journals/0067-0049/279/2/50/revision1/apjsade2d8t4_mrt.txt')
    kostov_new_eclipsing_binary_data_frame = ascii.read(kostov_new_eclipsing_binary_table_response.text,
                                                        format='mrt').to_pandas()
    kostov_known_eclipsing_binary_data_frame = ascii.read(kostov_known_eclipsing_binary_table_response.text,
                                                          format='mrt').to_pandas()
    tic_ids = list(set((kostov_new_eclipsing_binary_data_frame['TIC'].values.tolist() +
               kostov_known_eclipsing_binary_data_frame['TIC'].values.tolist())))
    return tic_ids


def metadata_data_frame_from_path_list(path_list: list[Path]) -> pd.DataFrame:
    tic_ids = []
    sectors = []
    paths = []
    tess_ffi_light_curve = TessFfiLightCurve()
    for path in path_list:
        tic_id, sector = tess_ffi_light_curve.get_tic_id_and_sector_from_file_path(path)
        tic_ids.append(tic_id)
        sectors.append(sector)
        paths.append(path)
    data_frame = pd.DataFrame({'tic_id': tic_ids, 'sector': sectors, 'relative_path': paths})
    return data_frame


def create_metadata_csv_file_for_path_list(path_list: list[Path], csv_file_path: Path):
    data_frame = metadata_data_frame_from_path_list(path_list)
    data_frame.to_csv(csv_file_path, index=False)


def create_split_datasets_metadata_csv_files_for_paths_list(path_list: list[Path], csv_path_base_file: Path):
    path_10_percent_lists: list[set[Path]] = [set() for _ in range(10)]
    data_frame_with_metadata = metadata_data_frame_from_path_list(path_list)
    data_frame_with_metadata['tic_id_count'] = data_frame_with_metadata.groupby('tic_id')['tic_id'].transform('count')
    data_frame_with_metadata = data_frame_with_metadata.sort_values(by='tic_id_count', ascending=False
                                                                    ).drop(columns=['tic_id_count'])
    data_frame_with_metadata = data_frame_with_metadata.reset_index(drop=True)
    while data_frame_with_metadata.shape[0] > 1:
        smallest_list_index = np.argmin([len(sub_list) for sub_list in path_10_percent_lists])
        next_tic_id = data_frame_with_metadata['tic_id'].iloc[0]
        sub_data_frame = data_frame_with_metadata[data_frame_with_metadata['tic_id'] == next_tic_id]
        path_10_percent_lists[smallest_list_index].update(map(Path, sub_data_frame['relative_path'].values.tolist()))
        data_frame_with_metadata = data_frame_with_metadata[data_frame_with_metadata['tic_id'] != next_tic_id]
    train_path_list = list(itertools.chain.from_iterable(path_10_percent_lists[:8]))
    validation_path_list = path_10_percent_lists[8]
    test_path_list = path_10_percent_lists[9]
    train_csv_path = csv_path_base_file.parent.joinpath(f'train_{csv_path_base_file.name}')
    validation_csv_path = csv_path_base_file.parent.joinpath(f'validation_{csv_path_base_file.name}')
    test_csv_path = csv_path_base_file.parent.joinpath(f'test_{csv_path_base_file.name}')
    create_metadata_csv_file_for_path_list(list(train_path_list), train_csv_path)
    create_metadata_csv_file_for_path_list(list(validation_path_list), validation_csv_path)
    create_metadata_csv_file_for_path_list(list(test_path_list), test_csv_path)


def refine_file_structure():
    dataset_root_directory = dataset_root_path
    dataset_root_directory.mkdir(exist_ok=True)
    dataset_data_directory = dataset_root_directory.joinpath('data')
    dataset_data_directory.mkdir(exist_ok=True)
    light_curve_directory = dataset_data_directory.joinpath('light_curves')
    light_curve_directory.mkdir(exist_ok=True)
    for csv_path in temporary_data_directory.glob('*.csv'):
        data_frame = pd.read_csv(csv_path, index_col=None)
        data_frame['file_name'] = data_frame['relative_path'].apply(relative_path_string_to_file_name)
        for relative_path_string in data_frame['relative_path']:
            relative_path = Path(relative_path_string)
            shutil.copy(relative_path, light_curve_directory.joinpath(relative_path.name))
        data_frame = data_frame.drop(columns=['relative_path'])
        data_frame.to_csv(dataset_data_directory.joinpath(csv_path.name))


def relative_path_string_to_file_name(relative_path_string: str) -> str:
    return Path(relative_path_string).name


def prepare_tess_transit_dataset():
    temporary_data_directory.mkdir(exist_ok=True, parents=True)
    dataset_root_path.mkdir(exist_ok=True, parents=True)
    dataset_light_curve_directory.mkdir(exist_ok=True, parents=True)
    with Path('tess_transit_dataset_preparation.log').open('w') as log_file:
        random = Random(0)
        transit_tic_ids = get_transit_tic_ids()
        random.shuffle(transit_tic_ids)
        non_transit_candidate_tic_ids = get_non_transit_candidate_tic_ids()
        random.shuffle(non_transit_candidate_tic_ids)
        eclipsing_binary_tic_ids = get_eclipsing_binary_tic_ids()
        random.shuffle(eclipsing_binary_tic_ids)
        download_spoc_light_curves_for_tic_ids(transit_tic_ids, dataset_light_curve_directory)
        download_spoc_light_curves_for_tic_ids(eclipsing_binary_tic_ids, dataset_light_curve_directory, limit=100_000)
        download_spoc_light_curves_for_tic_ids(non_transit_candidate_tic_ids, dataset_light_curve_directory, limit=500_000)
        light_curve_paths = list(dataset_light_curve_directory.glob('**/*.fits'))
        print(f'len paths: {len(light_curve_paths)}', file=log_file, flush=True)
        transit_tic_ids = set(get_transit_tic_ids())
        print(f'Transit len {len(transit_tic_ids)}', file=log_file, flush=True)
        non_transit_candidate_tic_ids = set(get_non_transit_candidate_tic_ids())
        print(f'Other len {len(non_transit_candidate_tic_ids)}', file=log_file, flush=True)
        eclipsing_binary_tic_ids = set(get_eclipsing_binary_tic_ids())
        print(f'EB len {len(eclipsing_binary_tic_ids)}', file=log_file, flush=True)
        transit_light_curve_paths = []
        eclipsing_binary_light_curve_paths = []
        other_light_curve_paths = []
        tess_ffi_light_curve = TessFfiLightCurve()
        for light_curve_index, light_curve_path in enumerate(light_curve_paths):
            tic_id, sector = tess_ffi_light_curve.get_tic_id_and_sector_from_file_path(light_curve_path)
            if tic_id in transit_tic_ids:
                transit_light_curve_paths.append(light_curve_path)
            elif tic_id in eclipsing_binary_tic_ids:
                eclipsing_binary_light_curve_paths.append(light_curve_path)
            elif tic_id in non_transit_candidate_tic_ids:
                other_light_curve_paths.append(light_curve_path)
            if light_curve_index % 10_000 == 0:
                print(light_curve_index)
        rng = Random(0)
        print(f'Length of transit TIC IDs: {len(transit_tic_ids)}', file=log_file, flush=True)
        rng.shuffle(transit_light_curve_paths)
        rng.shuffle(eclipsing_binary_light_curve_paths)
        rng.shuffle(other_light_curve_paths)
        eclipsing_binary_light_curve_paths = eclipsing_binary_light_curve_paths[:100_000]
        other_light_curve_paths = other_light_curve_paths[:500_000]
        create_split_datasets_metadata_csv_files_for_paths_list(transit_light_curve_paths,
                                                                temporary_data_directory.joinpath(
                                                                    'transit.csv'))
        create_split_datasets_metadata_csv_files_for_paths_list(eclipsing_binary_light_curve_paths,
                                                                temporary_data_directory.joinpath(
                                                                    'eclipsing_binary.csv'))
        create_split_datasets_metadata_csv_files_for_paths_list(other_light_curve_paths,
                                                                temporary_data_directory.joinpath(
                                                                    'other.csv'))
        refine_file_structure()


if __name__ == '__main__':
    main()
