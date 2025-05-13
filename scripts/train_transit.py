"""
TODO: Warning, this file had been sitting for a while with changes and was committed without confirming it's current
state.
"""

# from qusi.internal.haydros_model import Haydros
# from qusi.internal.torrin_model import Torrin, Torrin1, Torrin2, Torrin3, Torrin4, Torrin5, Torrin6, Torrin7, Torrin9

from qusi.internal.chyrin_model import Chyrin
from qusi.internal.train_session import train_session
from qusi.internal.train_logging_configuration import TrainLoggingConfiguration
from qusi.internal.train_system_configuration import TrainSystemConfiguration
from qusi.model import Hadryss
from qusi.session import TrainHyperparameterConfiguration
from qusi_evaluation.transit_dataset import get_train_dataset, get_validation_dataset, get_train_dataset_with_injection


def main():
    train_light_curve_dataset = get_train_dataset()
    validation_light_curve_dataset = get_validation_dataset()
    model = Hadryss.new(input_length=3500)
    train_system_configuration = TrainSystemConfiguration.new()
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=100, cycles=200, train_steps_per_cycle=1000, validation_steps_per_cycle=100)
    train_logging_configuration = TrainLoggingConfiguration.new(
        wandb_entity='ramjet',
        wandb_project='qusi_development',
        additional_log_dictionary={'run_name': f'{model.__class__.__name__}_transit_lightning'}
    )
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
                  model=model,
                  hyperparameter_configuration=train_hyperparameter_configuration,
                  system_configuration=train_system_configuration, logging_configuration=train_logging_configuration)


if __name__ == '__main__':
    main()
