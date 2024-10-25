from pathlib import Path

import torch

from qusi.data import FiniteStandardLightCurveDataset, LightCurveCollection
from qusi.model import Hadryss
from qusi.session import get_device, infer_session
from qusi_evaluation.microlensing_dataset import get_spoc_10_minute_ffi_light_curve_paths, \
    load_times_and_fluxes_from_tess_path


def main():
    infer_light_curve_collection = LightCurveCollection.new(
        get_paths_function=get_spoc_10_minute_ffi_light_curve_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_tess_path)

    test_light_curve_dataset = FiniteStandardLightCurveDataset.new(
       light_curve_collections=[infer_light_curve_collection])

    model = Hadryss.new()
    device = get_device()
    model.load_state_dict(torch.load('sessions/earthy-paper-21_latest_model.pt', map_location=device))
    confidences = infer_session(infer_datasets=[test_light_curve_dataset], model=model,
                                batch_size=100, device=device)[0]
    paths = list(get_spoc_10_minute_ffi_light_curve_paths())
    paths_with_confidences = zip(paths, confidences)
    sorted_paths_with_confidences = sorted(
        paths_with_confidences, key=lambda path_with_confidence: path_with_confidence[1], reverse=True)
    print(sorted_paths_with_confidences)


if __name__ == '__main__':
    main()
