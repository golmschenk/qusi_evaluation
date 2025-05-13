"""
TODO: Warning, this file had been sitting for a while with changes and was committed without confirming it's current
state.
"""

from pathlib import Path

from bokeh.io import save, output_file
from bokeh.model import Model
from bokeh.models import Div, Column
from bokeh.plotting import figure
from qusi_evaluation.transit_dataset import TransitDatasetTypeName, TransitDatasetClassName, \
    get_paths_for_class_and_dataset_type, load_times_and_fluxes_from_path


def main():
    for type_name in TransitDatasetTypeName:
        for class_name in TransitDatasetClassName:
            paths = get_paths_for_class_and_dataset_type(class_name, type_name)
            print(f'Paths for {class_name}, {type_name}: {len(paths)}')
            models: list[Model] = []
            for index in range(10):
                times, fluxes = load_times_and_fluxes_from_path(paths[index])
                figure_ = figure()
                figure_.line(x=times, y=fluxes, color='mediumblue', alpha=0.2)
                figure_.scatter(x=times, y=fluxes, color='mediumblue', alpha=0.5)
                models.append(figure_)
                models.append(Div(text=f'{paths[index]}'))
            column = Column(*models)
            title = f'check_{class_name}_{type_name}'
            file_path = Path(f'{title}.html')
            output_file(file_path)
            save(column, filename=file_path, title=title)



if __name__ == '__main__':
    main()