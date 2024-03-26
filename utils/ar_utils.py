import os

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from config.config import run_prefix


def get_coeff(model, train, cols, state_name, variant, run_prefix):
    coeff = pd.DataFrame(model.coef_, index=[' '.join(c) for c in train.columns],
                         columns=[' '.join(c) for c in cols])
    fig = px.bar(coeff,
                 orientation='h',
                 title=f'Coefficients for {state_name}',
                 barmode='group',
                 animation_frame=coeff.index)
    fig.write_html(f'{run_prefix}ar_models/figures/{variant}/coeff/coeff_{state_name}.html')
    coeff.to_csv(f"{run_prefix}ar_models/csv/{variant}/coeff/coeff_{state_name}.csv")


def get_states_dates():
    merged_df = pd.read_csv(f'{run_prefix}pipeline/csv/merged_regression_ready.csv')
    states = merged_df.reset_index(drop=True)
    state_names = list(states.state_name.unique())
    dates = list(set(list(states.date.unique())))
    return state_names, dates


def get_variant_states(inputs):
    variant = inputs[0] if len(inputs) == 1 else 'multi'
    os.makedirs(f"{run_prefix}ar_models/csv/{variant}", exist_ok=True)
    os.makedirs(f'{run_prefix}ar_models/figures/{variant}', exist_ok=True)
    print(f'Building models for {variant} input')
    merged_df = pd.read_csv(f'{run_prefix}pipeline/csv/merged_regression_ready.csv')
    states = merged_df.reset_index(drop=True)
    states = [
        (state_name, df.pivot(index='date', columns='occupation_name', values=inputs).sort_values(
            'date').interpolate().fillna(0)) for state_name, df in states.groupby('state_name')]
    return variant, states


def get_rmse_state_occ(pred, y_test, state_name, occ_name):
    return pd.DataFrame(mean_squared_error(pred, y_test, squared=False),
                        index=list(pred.columns)[0][1:] if type(
                            pred.columns) is pd.core.indexes.base.Index else pred.columns,
                        columns=[f'RMSE for {state_name}_{occ_name}'])


def get_rmse(pred, y_test, state_name, occ_names):
    return pd.DataFrame(mean_squared_error(pred, y_test, multioutput='raw_values', squared=False),
                        index=occ_names, columns=[f'RMSE for {state_name}'])


def get_values(data, col, variant):
    return data[col].values if variant == 'multi' else data


class ARLinearRegressionTry:

    def __init__(self, col, variant, n_jobs=-1):
        self.col = col
        self.variant = variant
        self.y_train = None
        self.X_train = None
        self.n_jobs = n_jobs
        self.reshape_size = None
        self.model = LinearRegression(n_jobs=self.n_jobs)

    def fit(self, X, y):
        self.X_train, self.y_train = X, get_values(y, self.col, self.variant)
        self.reshape_size = (-1, self.X_train.shape[1])

    def predict(self, X, y):
        preds = []
        for idx, (next_x, next_y) in enumerate(zip(X, y)):
            X_train = np.concatenate((self.X_train, X[:idx]))
            y_train = np.concatenate((self.y_train, y[:idx]))

            self.model.fit(X_train, y_train)
            preds.extend(self.model.predict(next_x.reshape(self.reshape_size)))

        return preds


class ARLinearRegression:

    def __init__(self, n_jobs=-1):
        self.y_train = None
        self.X_train = None
        self.n_jobs = n_jobs
        self.reshape_size = None
        # self.model = LinearRegression(alpha=0.2, normalize=True) # n_jobs=self.n_jobs
        self.model = LinearRegression(n_jobs=self.n_jobs)

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        self.reshape_size = (-1, self.X_train.shape[1])

    def predict(self, X, y):
        preds = []
        for idx, (next_x, next_y) in enumerate(zip(X, y)):
            X_train = np.concatenate((self.X_train, X[:idx]))
            y_train = np.concatenate((self.y_train, y[:idx]))
            self.model.fit(X_train, y_train)
            preds.extend(self.model.predict(next_x.reshape(self.reshape_size)))

        return preds
