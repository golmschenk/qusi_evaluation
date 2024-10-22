from qusi.model import Hadryss
from qusi.session import TrainHyperparameterConfiguration, train_session

from qusi_evaluation.slb_dataset import get_slb_train_dataset


def main():
    train_light_curve_dataset = get_slb_train_dataset()
    validation_light_curve_dataset = get_slb_train_dataset()
    model = Hadryss.new()
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=1000, cycles=200, train_steps_per_cycle=500, validation_steps_per_cycle=50)
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
                  model=model, hyperparameter_configuration=train_hyperparameter_configuration)


if __name__ == '__main__':
    main()
