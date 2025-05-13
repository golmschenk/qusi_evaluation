"""
TODO: Warning, this file had been sitting for a while with changes and was committed without confirming it's current
state.
"""

from pathlib import Path

import pandas as pd

from ramjet.data_interface.tess_data_interface import download_spoc_light_curves_for_tic_ids


def main():
    metadata_directory = Path('data/transit_evaluation')
    for metadata_csv_path in metadata_directory.glob('*.csv'):
        metadata_data_frame = pd.read_csv(metadata_csv_path, index_col=None)
        download_spoc_light_curves_for_tic_ids(metadata_data_frame['tic_id'],
                                               download_directory=Path('data/spoc_sector_27_to_55_light_curves'),
                                               sectors=list(range(27, 56)))


if __name__ == '__main__':
    main()
