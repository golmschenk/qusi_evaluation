"""
TODO: Warning, this file had been sitting for a while with changes and was committed without confirming it's current
state.
"""

from qusi.internal.chyrin_model import Chyrin
from qusi.internal.hadryss_model import HadryssMultiClassScoreEndModule
from qusi.internal.train_session import train_session
from qusi.internal.metric import CrossEntropyAlt, MulticlassAccuracyAlt, MulticlassAUROCAlt
from qusi.internal.train_logging_configuration import TrainLoggingConfiguration
from qusi.internal.train_system_configuration import TrainSystemConfiguration
from qusi.model import Hadryss
from qusi.session import TrainHyperparameterConfiguration
from qusi_evaluation.variable_team_simulations_dataset import get_train_dataset, get_validation_dataset


def main():
    train_light_curve_dataset = get_train_dataset()
    validation_light_curve_dataset = get_validation_dataset()
    number_of_classes = len(train_light_curve_dataset.standard_light_curve_collections)
    model = Hadryss.new(input_length=7000, end_module=HadryssMultiClassScoreEndModule.new(number_of_classes))
    loss_metric = CrossEntropyAlt()
    logging_metrics = [CrossEntropyAlt(), MulticlassAccuracyAlt(number_of_classes=number_of_classes),
                       MulticlassAUROCAlt(number_of_classes=number_of_classes)]
    train_system_configuration = TrainSystemConfiguration.new()
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=100, cycles=200, train_steps_per_cycle=100, validation_steps_per_cycle=10)
    train_logging_configuration = TrainLoggingConfiguration.new(additional_log_dictionary={'run_name': 'test'})
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
                  model=model, loss_metric=loss_metric, logging_metrics=logging_metrics,
                  hyperparameter_configuration=train_hyperparameter_configuration,
                  system_configuration=train_system_configuration, logging_configuration=train_logging_configuration)


if __name__ == '__main__':
    main()
