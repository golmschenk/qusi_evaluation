import pandas as pd
import torch
from qusi_evaluation.rich_microlensing_dataset import get_tess_light_curves_paths, get_rich_infer_dataset

from qusi.model import Hadryss
from qusi.session import get_device, infer_session


def main():
    model = Hadryss.new()
    device = get_device()
    model.load_state_dict(torch.load('sessions/worldly-tree-26_latest_model.pt', map_location=device))
    confidences = infer_session(infer_datasets=[get_rich_infer_dataset()], model=model,
                                batch_size=100, device=device)[0]
    paths = list(get_tess_light_curves_paths())
    paths_with_confidences = zip(paths, confidences)
    sorted_paths_with_confidences = sorted(
        paths_with_confidences, key=lambda path_with_confidence: path_with_confidence[1], reverse=True)
    data_frame = pd.DataFrame(data=sorted_paths_with_confidences, columns=['paths', 'confidences'])
    data_frame.to_csv('rich_infer_results.csv')


if __name__ == '__main__':
    main()
