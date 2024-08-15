from pathlib import Path

from qusi.experimental.application.tess import (
    download_spoc_light_curves_for_tic_ids,
    get_spoc_tic_id_list_from_mast,
)

spoc_sector_27_to_56_light_curve_directory = Path('data/spoc_sector_27_to_56_light_curves')


def main():
    print('Retrieving metadata...')
    spoc_target_tic_ids = get_spoc_tic_id_list_from_mast()
    sectors = list(range(27, 56))

    print('Downloading light curves...')
    download_spoc_light_curves_for_tic_ids(
        tic_ids=spoc_target_tic_ids,
        download_directory=spoc_sector_27_to_56_light_curve_directory,
        sectors=sectors)


if __name__ == '__main__':
    main()
