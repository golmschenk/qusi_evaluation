import subprocess
from pathlib import Path


def download_plasticc_dataset():
    data_directory = Path('data/plasticc')
    data_directory.mkdir(exist_ok=True, parents=True)
    subprocess.run(['zenodo_get', '2539456'], cwd=data_directory)

if __name__ == '__main__':
    download_plasticc_dataset()
