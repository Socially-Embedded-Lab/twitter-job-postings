import os
import string
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from patsy import dmatrices

from config.config import ARConfig

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

run_prefix = '' if os.getcwd().endswith('Labour-Market-Statistics') else '../'

variant_dict = {}

os.makedirs(f"{run_prefix}ar_models/csv", exist_ok=True)
os.makedirs(f"{run_prefix}ar_models/csv/pooled_effects", exist_ok=True)
os.makedirs(f"{run_prefix}ar_models/figures", exist_ok=True)
os.makedirs(f"{run_prefix}ar_models/figures/pooled_effects", exist_ok=True)

csv_prefix = f"{run_prefix}ar_models/csv/pooled_effects"
fig_prefix = f"{run_prefix}ar_models/figures/pooled_effects"

merged_df = pd.read_csv(f'{run_prefix}pipeline/csv/merged_regression_ready.csv', parse_dates=['date'])
states = merged_df.reset_index(drop=True)
shifted = []
dropped_occ = []
dropped_state = []
for (state_name, occupation_name), gdf in states.groupby(['state_name', 'occupation_name']):
    if occupation_name in dropped_occ or state_name in dropped_state:
        continue
    source = gdf[['date', 'state_name', 'occupation_name']]
    source[f'employment'] = gdf[f'employment{ARConfig.PREFIX}'].shift(0)
    source[f'job_offer'] = gdf[f'job_offer{ARConfig.PREFIX}'].shift(0)
    source['target'] = gdf[f'employment{ARConfig.PREFIX}'].shift(-1)
    source['target_date'] = gdf['date'].shift(-1)
    source = source.dropna()
    if len(source) >= 40:
        shifted.append(source)
states = pd.concat(shifted)

state_names = sorted(list(states.state_name.unique()))
occ_names = sorted(list(states.occupation_name.unique()))

dropped_df = pd.DataFrame(index=state_names, columns=occ_names).replace(np.NAN, False)

for state_name in tqdm(state_names):
    if state_name not in variant_dict:
        variant_dict[state_name] = []
        labels = []
        dates_index = []
    data = states[states['state_name'] == \
                  state_name].sort_values(by=['occupation_name', 'date']).set_index('date')
    train = pd.DataFrame(columns=data.columns)
    test = pd.DataFrame(columns=data.columns)
    # Group data by occupation and split into train and test
    for occupation, group in data.groupby('occupation_name'):
        train_group, test_group = train_test_split(group, train_size=0.7, shuffle=False)
        train = pd.concat([train, train_group])
        test = pd.concat([test, test_group])

    occ_state = sorted(list(data.occupation_name.unique()))
    for inputs in [['employment'], ['employment', 'job_offer']]:
        variant = inputs[0] if len(inputs) == 1 else 'job_offer_data'

        y_var_name = 'target'
        X_var_names = inputs

        pooled_y = train[y_var_name]
        pooled_X = train[X_var_names]

        pooled_X = sm.add_constant(pooled_X)
        pooled_olsr_model = sm.OLS(endog=pooled_y, exog=pooled_X)
        pooled_olsr_model_results = pooled_olsr_model.fit()

        if ARConfig.DEBUG:
            print(pooled_olsr_model_results.summary())

        preds = pooled_olsr_model_results.predict(sm.add_constant(test[X_var_names]))
        preds = pd.DataFrame(preds, columns=[y_var_name]).set_index(test.index)
        preds['occupation_name'] = test['occupation_name']

        rmse = {
            'occupation_name': [],
            f'rmse for {state_name}': []
        }

        for occupation_name, gdf in preds.groupby('occupation_name'):
            # rmse['occupation_name'].append(occ_translate[occupation_name])
            rmse['occupation_name'].append(occupation_name)
            rmse[f'rmse for {state_name}'].append(
                mean_squared_error(gdf[y_var_name], test[test['occupation_name'] == occupation_name][y_var_name],
                                   squared=False))

        rmse = pd.DataFrame(rmse).set_index('occupation_name')

        if variant == 'employment':
            ar_employment = preds.copy()
        else:
            ar_job_offer_data = preds.copy()
        variant_dict[state_name].append(rmse)

    ar_employment['label'] = 'employment'
    ar_job_offer_data['label'] = 'job_offer_data'
    train = train[['occupation_name', ARConfig.TARGET_VAR]]
    test = test[['occupation_name', ARConfig.TARGET_VAR]]
    train['label'] = 'input'
    test['label'] = 'input'

    combined_df = pd.concat([train, test, ar_employment, ar_job_offer_data])

    fig = px.line(combined_df,
                  symbol='occupation_name',
                  color='label',
                  markers=True, template="plotly_white",
                  title=f'Employment AR(1) for {state_name}',
                  labels=dict(date="Date", value="Employment rate (%)",
                              variable='State'))
    fig.write_html(f'{fig_prefix}/{ARConfig.TARGET_VAR}_rate_{state_name}.html')

boxes_states = []
all_data_state = []
for state_name in tqdm(variant_dict.keys(), total=len(variant_dict.keys())):
    df1, df2 = variant_dict[state_name][0], variant_dict[state_name][1]
    try:
        merged = pd.merge(df1, df2, left_index=True, right_index=True)
    except (IndexError, KeyError):
        continue
    merged.columns = ['employment', 'job_offer_data']
    if ARConfig.DEBUG:
        merged['job_offer_data_improved'] = merged['job_offer_data'] < merged['employment']
        print(
            f"For {state_name} Job Offer data improved a total of: {merged['job_offer_data_improved'].sum()}/{len(merged)}")
        print(f"with average RMSE of "
              f"{(merged[merged['job_offer_data_improved']]['employment'] - merged[merged['job_offer_data_improved']]['job_offer_data']).mean()}")
        print(
            f"employment is better with an average RMSE of "
            f"{(merged[~merged['job_offer_data_improved']]['job_offer_data'] - merged[~merged['job_offer_data_improved']]['employment']).mean()}")

    merged['rmse_difference'] = ((merged['employment'] / merged['job_offer_data']) - 1) * 100
    merged.to_csv(f"{csv_prefix}/merged_data_for_{state_name}.csv")
    box_plots = pd.DataFrame(index=df1.index)
    box_plots[state_name] = merged['rmse_difference']
    all_data = pd.DataFrame(index=list(df1.index) * 2)
    all_data['state_name'] = state_name
    all_data['label'] = [label for label in ['employment', 'job_offer_data'] for _ in range(len(df1.index))]
    all_data['value'] = np.append(merged['employment'], merged['job_offer_data'])
    boxes_states.append(box_plots)
    all_data_state.append(all_data)

box_plots = pd.concat(boxes_states, axis=1).replace(np.inf, np.NAN)
all_data = pd.concat(all_data_state).reset_index().rename({'index': 'occupation_name'}, axis=1)
all_data.to_csv(f'{csv_prefix}/Joint RMSE for all data.csv')

fig = px.bar(all_data, x='occupation_name', y='value',
             color='label', title='Joint RMSE for all data',
             animation_frame='state_name', barmode='group',
             labels=dict(state_name='State Name', label='Input Data'))
fig.write_html(f'{fig_prefix}/Joint RMSE for all data.html')

fig = px.bar(box_plots.mean(axis=1).dropna(),
             orientation='h',
             title='Mean Improvement of RMSE(%) by Occupation using job offers data',
             labels=dict(index='Occupation Name', value='RMSE(%) Improvement'))
fig.update_layout(showlegend=False)
fig.write_html(f'{fig_prefix}/{ARConfig.TARGET_VAR}_rate_RMSE_improve_occ.html')

print(f'Mean Improvement of RMSE(%) by Occupation using job offers data')
print(box_plots.mean(axis=1).dropna())
box_plots.mean(axis=1).dropna().to_csv(f'{csv_prefix}/{ARConfig.TARGET_VAR}_rate_RMSE_improve_occ.csv')

fig = px.bar(box_plots.mean().dropna(),
             orientation='h',
             title='Mean Improvement of RMSE(%) by State using job offers data',
             labels=dict(index='State Name', value='RMSE(%) Improvement'))
fig.update_layout(showlegend=False)
fig.write_html(f'{fig_prefix}/{ARConfig.TARGET_VAR}_rate_RMSE_improve_state.html')

print(f'Mean Improvement of RMSE(%) by State using job offers data')
print(box_plots.mean().dropna())
box_plots.mean().dropna().to_csv(f'{csv_prefix}/{ARConfig.TARGET_VAR}_rate_RMSE_improve_state.csv')

fig = px.bar(box_plots,
             orientation='h',
             title='RMSE(%) Improvement using job offer data',
             labels=dict(index='State Name', value='RMSE(%) Improvement', variable='State',
                         occupation_name='Occupation Name'))
fig.write_html(f'{fig_prefix}/RMSE Improvement using job offer data.html')

box_plots.to_csv(f'{csv_prefix}/RMSE Improvement using job offer data.csv')

print(f'Average(%) improvement across Occupation Name and State is: {box_plots.mean().mean()}')
