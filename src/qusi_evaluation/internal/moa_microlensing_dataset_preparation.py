from pathlib import Path

import numpy as np
import pandas as pd
from merida.lightcurves_cls import LightCurvesNExSciURL, Metadata

dataset_root_path = Path('data/general_light_curve_benchmark_dataset_collection_moa_microlensing_dataset')
dataset_light_curve_directory = dataset_root_path.joinpath('light_curves')


def prepare_moa_dataset():
    download_light_curves()
    create_metadata_splits()


def download_light_curve(light_curve_name):
    light_curve = LightCurvesNExSciURL(lightcurve_name_=light_curve_name, lightcurve_class_='')
    light_curve.save_lightcurve_from_url_as_feather(path_to_save=str(dataset_light_curve_directory) + '/')


def download_light_curves():
    candidate_tags = ['c', 'cf', 'cp', 'cw', 'cs', 'cb']
    non_candidate_tags = ['v', 'n', 'nr', 'm', 'j']
    no_tag_tags = ['no_tag', '', None, np.nan]
    dataset_light_curve_directory.mkdir(exist_ok=True, parents=True)
    metadata_data_frame = prepare_full_metadata_data_frame()
    candidate_metadata_data_frame = metadata_data_frame[metadata_data_frame['tag'].isin(candidate_tags)]
    for light_curve_name in candidate_metadata_data_frame['lightcurve_name']:
        print(f'Downloading {light_curve_name}...')
        # download_light_curve(light_curve_name)
    for non_candidate_tag in non_candidate_tags:
        non_candidate_metadata_data_frame = metadata_data_frame[metadata_data_frame['tag'].isin([non_candidate_tag])]
        non_candidate_metadata_data_frame = non_candidate_metadata_data_frame.sample(frac=1.0, random_state=0)
        for light_curve_name in non_candidate_metadata_data_frame['lightcurve_name']:
            print(f'Downloading {light_curve_name}...')
            # download_light_curve(light_curve_name)
    no_tag_limit = 100_000
    no_tag_metadata_data_frame = metadata_data_frame[metadata_data_frame['tag'].isin(no_tag_tags)]
    no_tag_metadata_data_frame = no_tag_metadata_data_frame.sample(frac=1.0, random_state=0)
    no_tag_metadata_data_frame = no_tag_metadata_data_frame.head(no_tag_limit)
    for light_curve_name in no_tag_metadata_data_frame['lightcurve_name']:
        print(f'Downloading {light_curve_name}...')
        # download_light_curve(light_curve_name)


def prepare_full_metadata_data_frame() -> pd.DataFrame:
    metadata_data_frame = Metadata().dataframe
    metadata_data_frame = metadata_data_frame.drop('tag', axis=1)
    candlist_data_frame = prepare_candidate_data_frame()
    metadata_data_frame = pd.merge(metadata_data_frame, candlist_data_frame, how='left',
                                   on=['field', 'chip', 'subframe', 'id'])
    metadata_data_frame.insert(4, 'tag', metadata_data_frame.pop('tag'))
    return metadata_data_frame


def prepare_candidate_data_frame():
    candlist_data_frame = pd.read_csv(Path('data/candlist_2023Oct12.txt'), comment='#',
                                      sep=r'\s+', names=list(range(33)))

    def extract_field_number(field_id: str) -> int:
        return int(field_id.replace('gb', ''))

    candlist_data_frame['field'] = candlist_data_frame[0].apply(extract_field_number)
    candlist_data_frame['chip'] = candlist_data_frame[2].astype(np.int64)
    candlist_data_frame['subframe'] = candlist_data_frame[3].astype(np.int64)
    candlist_data_frame['id'] = candlist_data_frame[4].astype(np.int64)
    candlist_data_frame['tag'] = candlist_data_frame[5]
    candlist_data_frame = candlist_data_frame.filter(['field', 'chip', 'subframe', 'id', 'tag'])
    return candlist_data_frame


def create_metadata_splits():
    metadata_data_frame = prepare_full_metadata_data_frame()
    metadata_data_frame = metadata_data_frame.sample(frac=1.0, random_state=0)
    file_names: list[str] = []
    tags: list[str] = []
    for index, row in metadata_data_frame.iterrows():
        path = dataset_light_curve_directory.joinpath(
            f'gb{row["field"]}-R-{row["chip"]}-{row["subframe"]}-{row["id"]}.feather')
        if path.exists():
            file_names.append(path.name)
            if row['tag'] in ['no_tag', '', None, np.nan]:
                tags.append('no_tag')
            else:
                tags.append(str(row['tag']))
    unique_tags = list(set(tags))
    tag_count_dictionary = {tag: 0 for tag in unique_tags}
    file_name_split_lists: list[list[str]] = [[] for _ in range(10)]
    tags_split_lists: list[list[str]] = [[] for _ in range(10)]
    for file_name, tag in zip(file_names, tags):
        split_to_insert_to = tag_count_dictionary[tag] % 10
        file_name_split_lists[split_to_insert_to].append(file_name)
        tags_split_lists[split_to_insert_to].append(tag)
        tag_count_dictionary[tag] += 1
    test_dataset = pd.DataFrame({'file_name': file_name_split_lists[0], 'tag': tags_split_lists[0]})
    validate_dataset = pd.DataFrame({'file_name': file_name_split_lists[1], 'tag': tags_split_lists[1]})
    train_dataset = pd.DataFrame({
        'file_name': file_name_split_lists[2] + file_name_split_lists[3] + file_name_split_lists[4] +
                     file_name_split_lists[5] + file_name_split_lists[6] + file_name_split_lists[7] +
                     file_name_split_lists[8] + file_name_split_lists[9],
        'tag': tags_split_lists[2] + tags_split_lists[3] + tags_split_lists[4] +
               tags_split_lists[5] + tags_split_lists[6] + tags_split_lists[7] +
               tags_split_lists[8] + tags_split_lists[9]})
    test_dataset.to_csv(dataset_root_path.joinpath('test_metadata.csv'), index=False)
    validate_dataset.to_csv(dataset_root_path.joinpath('validate_metadata.csv'), index=False)
    train_dataset.to_csv(dataset_root_path.joinpath('train_metadata.csv'), index=False)
