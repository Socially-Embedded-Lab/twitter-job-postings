import os
import sys
from glob import glob


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config.config import JobTitleConfig
from utils.pipeline_utils import *


def plot_city(df):
    fig = px.scatter_mapbox(df, lat='lat', lon='lon',
                            hover_name='job_titles',
                            color='category_label',
                            mapbox_style="carto"
                                         "-positron",
                            size_max=40, zoom=3,
                            hover_data={'address': True,
                                        'input_text': True,
                                        'category_label': True,
                                        'original_address': True,
                                        'user_id': False,
                                        'tweet_id': False,
                                        'lat': False,
                                        'lon': False,
                                        'created_at': False,
                                        'created_at_str': False},
                            animation_frame='created_at_str',
                            labels=dict(created_at_str='Date',
                                        original_address='Extracted Address',
                                        category_label='Occupation Group',
                                        input_text='Clean Tweet',
                                        job_titles='Extracted Job Title',
                                        address='Address'),
                            color_discrete_sequence=px.colors.qualitative.Light24)
    fig.update_layout(legend_title_text='Occupation Group')
    fig.write_html(f'{pipeline_path}/figures/granular_map.html')


def agg_city(df):
    category_count = df.groupby(['created_at', 'address', 'lat', 'lon', 'category_label'])['categories'].count().sort_values(
        ascending=False).reset_index()
    category_count.columns = ['created_at', 'address', 'lat', 'lon', 'category_label', 'count']
    category_count = category_count[['created_at', 'address', 'category_label', 'lat', 'lon', 'count']]

    category_count['year'] = category_count['created_at'].apply(lambda d: d.year)
    category_count['month'] = category_count['created_at'].apply(lambda d: d.month)
    category_count = category_count.sort_values(by=['year', 'month'])
    category_count['created_at_str'] = category_count['created_at'].apply(lambda ca: str(ca)[:10])

    fig = px.scatter_mapbox(category_count, lat='lat', lon='lon',
                            hover_name='address', size='count',
                            mapbox_style="carto"
                                         "-positron",
                            color='category_label',
                            hover_data={'address': True,
                                        'category_label': True,
                                        'lat': False,
                                        'lon': False,
                                        'created_at_str': False,
                                        'count': True},
                            labels=dict(created_at_str='Date',
                                        address='Address',
                                        category_label='Occupation Group'),
                            animation_frame='created_at_str', zoom=3)
    fig.update_layout(legend_title_text='Occupation Group')
    fig.write_html(f'{pipeline_path}/figures/category_count_map.html')
    category_count.to_csv(f'{pipeline_path}/csv/3M_category_count.csv', index=False)


pipeline_path = PipeLineConfig.folder
files = glob(f'{pipeline_path}/csv/merged/*.csv')
merged_df = pd.read_csv(files[0]) if len(files) == 1 else pd.concat([pd.read_csv(f) for f in files])

try:
    merged_df = merged_df.drop(['category_label.1', 'original_address.1'], axis=1)
except:
    print(f'missing columns, ignore!')
merged_df.to_csv(f'{pipeline_path}/csv/sample_merged.csv', index=False)

isco_count = len(merged_df.category.unique())
digit_count = 1 if isco_count <= 10 else 2

labels_map = pd.read_csv(f'{PipeLineConfig.JOB_TITLES_DIR}/labels_map.csv', index_col='category')
num_labels = len(labels_map)

official_state = pd.read_csv(f'{run_prefix}data/csv/by_state_by_occupation_2_digit_new.csv')
official_state = official_state.dropna(subset=['occupation_name']).reset_index(drop=True)

if 'sub_major' not in PipeLineConfig.JOB_TITLES_DIR:
    official_state['occupation_code'] = official_state['occupation_code'] // 10

if JobTitleConfig.merge:
    category_mapping = pd.read_csv(JobTitleConfig.category_mapping, index_col=0)
    official_state['occupation_code'] = category_mapping.loc[official_state['occupation_code'].map(int).map(str).apply(lambda code: int(code.zfill(2)[:digit_count]))].values
official_state[['occupation_code', 'example', 'occupation_name']] = labels_map.loc[official_state['occupation_code']].reset_index().values

state_emp = get_data_from_stats(official_state, '#_employed_people')

merged_df = merged_df[merged_df['created_at'].map(str).apply(lambda d: len(d.split('-'))) == 3].reset_index(
    drop=True)
merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
merged_df['year'] = merged_df['created_at'].apply(lambda d: d.year)
merged_df['month'] = merged_df['created_at'].apply(lambda d: d.month)
merged_df = merged_df.sort_values(by=['year', 'month'])
merged_df['created_at_str'] = merged_df['created_at'].apply(lambda ca: str(ca)[:10])
merged_df = merged_df.dropna(subset=['category_label']).reset_index(drop=True)

if False:
    plot_city(merged_df)
    agg_city(merged_df)

merged_df.to_csv(f'{run_prefix}pipeline/csv/merged.csv')

merged_df = pd.read_csv(f'{run_prefix}pipeline/csv/merged.csv', parse_dates=['created_at'])
# merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])

plot_df = merged_df[['created_at', 'category_label', 'address', 'lat', 'lon']]
plot_df = plot_df.sort_values(by=['created_at'])
plot_df = plot_df[plot_df.address.apply(lambda add: len(str(add).split(', ')) in (2, 3))]

plot_df['city'] = plot_df.address.apply(lambda add: add.split(', ')[0])
plot_df['state'] = plot_df.address.apply(lambda add: add.split(', ')[-2])

plot_df = plot_df.rename({'created_at': 'date'}, axis=1)
plot_df['year'] = plot_df['date'].apply(lambda d: d.year)
plot_df['month'] = plot_df['date'].apply(lambda d: d.month)

state_df = plot_df.groupby(['year', 'month', 'state', 'category_label']).category_label.agg(
    ['count']).reset_index()
state_df['rate'] = state_df.groupby(['year', 'month', 'state'])['count'].apply(
    lambda x: 100 * x / float(x.sum())).reset_index()['count']
state_df['rate2'] = state_df.groupby(['year', 'month'])['count'].apply(
    lambda x: 100 * x / float(x.sum())).reset_index()['count']

state_df['date'] = get_date_from_df(state_df)
state_df = state_df.rename({'count': 'value', 'category_label': 'occupation_name', 'state': 'state_name'}, axis=1)

state_emp = state_emp.groupby(['year', 'month', 'state_name', 'occupation_name']).value.agg(
    ['sum']).reset_index()
state_emp['rate'] = state_emp.groupby(['year', 'month', 'state_name'])['sum'].apply(
    lambda x: 100 * x / float(x.sum())).reset_index()['sum']
state_emp['rate2'] = state_emp.groupby(['year', 'month'])['sum'].apply(
    lambda x: 100 * x / float(x.sum())).reset_index()['sum']
state_emp['date'] = get_date_from_df(state_emp)
state_emp = state_emp.rename({'sum': 'value'}, axis=1)

# plot_and_save_line(df=state_df, sort_values=['state_name', 'occupation_name'], x='date', y='value',
#                    color='occupation_name',
#                    title='Monthly job offer rate (%) per state',
#                    labels=dict(date="Date", value="Job offer (%)", occupation_name="Occupation Name"),
#                    name='fig_state_offer')
#
# plot_and_save_line(state_emp, sort_values=['state_name', 'occupation_name'], x='date', y='value',
#                    color='occupation_name',
#                    title='Monthly Employment rate by state',
#                    labels=dict(date="Date", value="Employment Rate (%)", occupation_name="Occupation Name"),
#                    name='fig_state_emp')

state_df['source'] = 'job_offer'
state_emp['source'] = 'employment'

state_df = state_df.drop(['year', 'month'], axis=1)
state_emp = state_emp.drop(['year', 'month'], axis=1)

combined_df = pd.concat([state_df, state_emp])
# plot_and_save_line(combined_df, sort_values=['state_name', 'occupation_name'], x='date',
#                    y='value',
#                    color='source', symbol="occupation_name",
#                    title='Combined Monthly data per state',
#                    labels=dict(date="Date", value="Job offer / Employment Rate (%)",
#                                occupation_name="Occupation Name",
#                                source='Value Source'), name='fig_state_emp_combined')

state_df = state_df.rename({'value': 'job_offer'}, axis=1)
state_df = state_df.rename({'rate': 'job_offer_rate'}, axis=1)
state_df = state_df.rename({'rate2': 'job_offer_rate_overall'}, axis=1)
state_emp = state_emp.rename({'value': 'employment'}, axis=1)
state_emp = state_emp.rename({'rate': 'employment_rate'}, axis=1)
state_emp = state_emp.rename({'rate2': 'employment_rate_overall'}, axis=1)

merged_df = pd.merge(state_df, state_emp, on=['state_name', 'occupation_name', 'date']).drop(
    ['source_x', 'source_y'],
    axis=1)

merged_df = merged_df[
    ['date', 'state_name', 'occupation_name', 'job_offer', 'employment', 'job_offer_rate', 'employment_rate',
     'job_offer_rate_overall', 'employment_rate_overall']]
merged_df = merged_df.dropna().reset_index(drop=True)
merged_df = merged_df.sort_values(by=['state_name', 'date', 'occupation_name'])
merged_df.to_csv(f'{run_prefix}pipeline/csv/merged_regression_ready.csv', index=False)

files = glob(f'{run_prefix}pipeline/csv/execution_times_*.csv')
df = pd.concat([pd.read_csv(f, index_col=0) for f in files], axis=1)
df.mean(axis=1).to_csv('execution_times.csv')
