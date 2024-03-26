import os
import numpy as np


def makedirs(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)


def plot_error_bars(df, title, xlabel, ylabel, y_ticks_labels, color=None):
    if color is None:
        color = ["steelblue", "seagreen"]
    ax = df.plot(kind="barh", y="mean", legend=False, title=title, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticklabels(y_ticks_labels)
    for key, spine in ax.spines.items():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False)
    ax.errorbar(df["mean"], df.index, xerr=df["double_std"],
                linewidth=1.5, color="black", alpha=0.4, capsize=4)


def normalize_data(series):
    return series / series.sum() * 100


def normalized_share(series1, series2):
    return normalize_data(series1) - normalize_data(series2)


def normalized_col(df_to_norm, column1, column2):
    return normalized_share(df_to_norm[column1], df_to_norm[column2])


def double_std(array):
    return np.std(array) * 2
