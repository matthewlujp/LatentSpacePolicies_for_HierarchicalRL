import os
import shutil
import cv2
import toml
import numpy as np
from collections import namedtuple
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line


# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='epoch'):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
        ys = np.asarray(ys_population, dtype=np.float32)
        ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
        trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
        trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
        trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
        trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
        trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
        data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
    else:
        data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]

    plotly.offline.plot({
      'data': data,
      'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)


def multiple_lineplot(xs, ys, title, line_names=None, path='', xaxis='epoch'):
    """Allow multiple time series to be plotted.
    """
    xs = np.array(xs, dtype=np.int)
    ys = np.array(ys, dtype=np.float32)
    assert xs.shape[0] == ys.shape[0], "{} != {}".format(xs.shape[0], ys.shape[0])
    assert len(ys.shape) == 2, ys.shape
    line_number = ys.shape[1]
    collors = ['rgb({},0,{})'.format(max(255 - i*50, 0), min(i*50, 255)) for i in range(line_number)]
    if line_names is not None:
        assert len(line_names) == line_number, "number of line_names {} does not match with line number {}, {}".format(len(line_names), line_number, line_names)
    else:
        line_names = [str(i) for i in range(line_number)]
    data = [Scatter(x=xs, y=v, line=Line(color=collors[i]), name=line_names[i]) for i, v in enumerate(ys.T)]
    plotly.offline.plot({
      'data': data,
      'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)


def write_video(frames, title, path=''):
    frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()


def load_toml_config(config_filepath):
    conf_d = toml.load(open(config_filepath))
    return namedtuple('Config', conf_d.keys())(*conf_d.values())


def create_save_dir(save_dir, replace_if_test=True):
    if replace_if_test:
        # Check if saving directory is valid
        if "test" in save_dir and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(save_dir):
            raise ValueError("Directory {} already exists.".format(save_dir))

    # Create save dir
    os.makedirs(save_dir)
    data_dir = os.path.join(save_dir, 'data')
    os.makedirs(data_dir)
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir)

    return data_dir, log_dir
