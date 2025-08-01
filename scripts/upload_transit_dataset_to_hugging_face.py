from pathlib import Path

from huggingface_hub import upload_large_folder


def upload_transit_dataset_to_hugging_face():
    upload_large_folder(
        repo_id='general_light_curve_benchmark_dataset_collection_tess_transiting_planet_dataset',
        folder_path='general_light_curve_benchmark_dataset_collection_tess_transiting_planet_dataset',
        repo_type='dataset',
    )

if __name__ == '__main__':
    upload_transit_dataset_to_hugging_face()