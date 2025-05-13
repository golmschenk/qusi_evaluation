"""
TODO: Warning, this file had been sitting for a while with changes and was committed without confirming it's current
state.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from merida.lightcurves_cls import Metadata, LightCurvesNExSciURL


def download_lightcurve(light_curve_name):
    light_curve = LightCurvesNExSciURL(lightcurve_name_=light_curve_name, lightcurve_class_='')
    light_curve.save_lightcurve_from_url_as_feather(path_to_save='data/moa_light_curves')


def prepare_moa_microlensing_dataset():
    candidate_tags = ['c', 'cf', 'cp', 'cw', 'cs', 'cb']
    non_candidate_tags = ['v', 'n', 'nr', 'm', 'j']
    no_tag_tags = ['no_tag', '', None, np.nan]
    metadata_data_frame = Metadata().dataframe
    metadata_data_frame = metadata_data_frame.drop('tag', axis=1)
    candlist_data_frame = prepare_candidate_data_frame()
    metadata_data_frame = pd.merge(metadata_data_frame, candlist_data_frame, how='left', on=['field', 'chip', 'subframe', 'id'])
    metadata_data_frame.insert(4, 'tag', metadata_data_frame.pop('tag'))
    # Download all candidates.
    candidate_metadata_data_frame = metadata_data_frame[metadata_data_frame['tag'].isin(candidate_tags)]
    for light_curve_name in candidate_metadata_data_frame['lightcurve_name']:
        print(f'Downloading {light_curve_name}...')
        download_lightcurve(light_curve_name)
    non_candidate_limit = 3000
    for non_candidate_tag in non_candidate_tags:
        non_candidate_metadata_data_frame = metadata_data_frame[metadata_data_frame['tag'].isin([non_candidate_tag])]
        non_candidate_metadata_data_frame = non_candidate_metadata_data_frame.sample(frac=1.0, random_state=0)
        non_candidate_metadata_data_frame = non_candidate_metadata_data_frame.head(non_candidate_limit)
        for light_curve_name in non_candidate_metadata_data_frame['lightcurve_name']:
            print(f'Downloading {light_curve_name}...')
            download_lightcurve(light_curve_name)
    no_tag_limit = 100_000
    no_tag_metadata_data_frame = metadata_data_frame[metadata_data_frame['tag'].isin([no_tag_tags])]
    no_tag_metadata_data_frame = no_tag_metadata_data_frame.sample(frac=1.0, random_state=0)
    no_tag_metadata_data_frame = no_tag_metadata_data_frame.head(no_tag_limit)
    for light_curve_name in no_tag_metadata_data_frame['lightcurve_name']:
        print(f'Downloading {light_curve_name}...')
        download_lightcurve(light_curve_name)


def prepare_candidate_data_frame():
    candlist_data_frame = pd.read_csv(Path('/Users/golmschenk/Code/merida/data/candlist_2023Oct12.txt'), comment='#',
                                     sep='\s+', names=list(range(33)))
    def extract_field_number(field_id: str) -> int:
        return int(field_id.replace('gb', ''))

    candlist_data_frame['field'] = candlist_data_frame[0].apply(extract_field_number)
    candlist_data_frame['chip'] = candlist_data_frame[2].astype(np.int64)
    candlist_data_frame['subframe'] = candlist_data_frame[3].astype(np.int64)
    candlist_data_frame['id'] = candlist_data_frame[4].astype(np.int64)
    candlist_data_frame['tag'] = candlist_data_frame[5]
    candlist_data_frame = candlist_data_frame.filter(['field', 'chip', 'subframe', 'id', 'tag'])
    return candlist_data_frame


if __name__ == '__main__':
    prepare_moa_microlensing_dataset()
