from qusi.model import Hadryss
from qusi.session import TrainHyperparameterConfiguration, train_session
from qusi_evaluation.rich_microlensing_dataset import get_rich_train_dataset


def main():
    train_light_curve_dataset = get_rich_train_dataset()
    validation_light_curve_dataset = get_rich_train_dataset()
    for injectable in train_light_curve_dataset.injectable_light_curve_collections:
        print(len(list(injectable.path_getter.get_paths())), flush=True)
    model = Hadryss.new()
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=100, cycles=200, train_steps_per_cycle=100, validation_steps_per_cycle=50)
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
                  model=model, hyperparameter_configuration=train_hyperparameter_configuration)


if __name__ == '__main__':
    main()
