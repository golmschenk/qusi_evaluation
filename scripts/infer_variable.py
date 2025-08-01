import torch
import pandas as pd
from qusi.experimental.metric import CrossEntropyAlt, MulticlassAccuracyAlt, MulticlassAUROCAlt
from qusi.internal.hadryss_model import Hadryss, HadryssMultiClassScoreEndModule

from qusi.session import finite_datasets_test_session, get_device
from qusi_evaluation.variable_team_simulations_dataset import get_train_dataset, get_test_dataset


def main():
    test_light_curve_dataset = get_test_dataset()
    number_of_classes = len(['CEP', 'DSCT', 'ECL', 'ELL', 'HB', 'LPV', 'RRLYR', 'T2CEP'])
    model = Hadryss.new(input_length=7000, end_module=HadryssMultiClassScoreEndModule(number_of_classes=number_of_classes))
    # model.load_state_dict(torch.load('/home/abhina/Astroproject/sessions/defiant-dukat-105_latest_model.pt'))
    metric_functions = [CrossEntropyAlt(), MulticlassAccuracyAlt(number_of_classes=number_of_classes),
                        MulticlassAUROCAlt(number_of_classes=number_of_classes)]
    results = finite_datasets_test_session(test_datasets=[test_light_curve_dataset], model=model,
                                           metric_functions=metric_functions, batch_size=100)
    results_df = pd.DataFrame(results)
    results_df.to_csv("test_results101.csv", index=False)


if __name__ == '__main__':
    main()