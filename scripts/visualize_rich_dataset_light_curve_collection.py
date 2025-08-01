from pathlib import Path

import pandas as pd
from bokeh.io import show, save
from bokeh.models import Column
from bokeh.plotting import figure

from qusi.experimental.application.tess import TessMissionLightCurve


def main():
    figures = []
    data_frame = pd.read_csv('results_path.csv')
    for light_curve_path in data_frame['path'].values[:100]:
        light_curve = TessMissionLightCurve.from_path(light_curve_path)
        light_curve_figure = figure(x_axis_label='Time (BTJD)', y_axis_label='Flux')
        light_curve_figure.scatter(x=light_curve.times, y=light_curve.fluxes)
        light_curve_figure.line(x=light_curve.times, y=light_curve.fluxes, line_alpha=0.3)
        light_curve_figure.sizing_mode = 'stretch_width'
        figures.append(light_curve_figure)
    column = Column(*figures)
    save(column, 'results.html')


if __name__ == '__main__':
    main()
