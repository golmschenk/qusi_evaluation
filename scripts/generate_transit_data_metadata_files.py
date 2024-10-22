import random
from pathlib import Path

import pandas as pd

from ramjet.data_interface.tess_data_interface import get_spoc_tic_id_list_from_mast
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface, ToiColumns
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve
from qusi_evaluation.download_spoc_sector_27_to_56_light_curve_data import spoc_sector_27_to_55_light_curve_directory


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
        (tess_toi_data_interface.toi_dispositions[ToiColumns.disposition.value] == 'CP') &
        # Less than a TESS sector.
        (tess_toi_data_interface.toi_dispositions[ToiColumns.transit_period__days.value] <= 27) &
        # Greater than a smaller number to remove bad data points.
        (tess_toi_data_interface.toi_dispositions[ToiColumns.transit_period__days.value] >= 0.001)
        ][ToiColumns.tic_id.value]
    return positive_tic_ids.values


def get_eclipsing_binary_tic_ids():
    eclipsing_binaries_data_frame = pd.read_csv('data/TESS_EB_catalog_31Aug.csv')
    tic_ids = eclipsing_binaries_data_frame['ID'].values
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
    path_list_length = len(path_list)
    percent_80_index = int(path_list_length * 0.8)
    percent_90_index = int(path_list_length * 0.9)
    train_path_list = path_list[:percent_80_index]
    validation_path_list = path_list[percent_80_index:percent_90_index]
    test_path_list = path_list[percent_90_index:]
    train_csv_path = csv_path_base_file.parent.joinpath(f'train_{csv_path_base_file.name}')
    validation_csv_path = csv_path_base_file.parent.joinpath(f'validation_{csv_path_base_file.name}')
    test_csv_path = csv_path_base_file.parent.joinpath(f'test_{csv_path_base_file.name}')
    create_metadata_csv_file_for_path_list(train_path_list, train_csv_path)
    create_metadata_csv_file_for_path_list(validation_path_list, validation_csv_path)
    create_metadata_csv_file_for_path_list(test_path_list, test_csv_path)


def main():
    print(f'mark-1')
    light_curve_paths = list(spoc_sector_27_to_55_light_curve_directory.glob('**/*.fits'))
    print(f'len paths: {len(light_curve_paths)}')
    transit_tic_ids = get_transit_tic_ids()
    print(f'Transit len {len(transit_tic_ids)}')
    non_transit_candidate_tic_ids = get_non_transit_candidate_tic_ids()
    print(f'Other len {len(non_transit_candidate_tic_ids)}')
    eclipsing_binary_tic_ids = get_eclipsing_binary_tic_ids()
    print(f'EB len {len(eclipsing_binary_tic_ids)}')
    transit_light_curve_paths = []
    eclipsing_binary_light_curve_paths = []
    other_light_curve_paths = []
    tess_ffi_light_curve = TessFfiLightCurve()
    print(f'mark0')
    for light_curve_path in light_curve_paths:
        tic_id, sector = tess_ffi_light_curve.get_tic_id_and_sector_from_file_path(light_curve_path)
        if tic_id in transit_tic_ids:
            transit_light_curve_paths.append(light_curve_path)
        elif tic_id in eclipsing_binary_tic_ids:
            eclipsing_binary_light_curve_paths.append(light_curve_path)
        elif tic_id in non_transit_candidate_tic_ids:
            other_light_curve_paths.append(light_curve_path)
    print(f'mark1')
    rng = random.Random(0)
    print(f'len transit paths: {len(transit_tic_ids)}')
    rng.shuffle(transit_light_curve_paths)
    rng.shuffle(eclipsing_binary_light_curve_paths)
    rng.shuffle(other_light_curve_paths)
    eclipsing_binary_light_curve_paths = eclipsing_binary_light_curve_paths[:100_000]
    other_light_curve_paths = other_light_curve_paths[:500_000]
    transit_evaluation_metadata_directory = Path('data/transit_evaluation')
    transit_evaluation_metadata_directory.mkdir(parents=True, exist_ok=True)
    print(f'mark2')
    create_split_datasets_metadata_csv_files_for_paths_list(transit_light_curve_paths, transit_evaluation_metadata_directory.joinpath('transit.csv'))
    create_split_datasets_metadata_csv_files_for_paths_list(eclipsing_binary_light_curve_paths, transit_evaluation_metadata_directory.joinpath('eclipsing_binary.csv'))
    create_split_datasets_metadata_csv_files_for_paths_list(other_light_curve_paths, transit_evaluation_metadata_directory.joinpath('other.csv'))


if __name__ == '__main__':
    main()
