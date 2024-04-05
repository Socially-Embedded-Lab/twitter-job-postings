import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio
import matplotlib.colors as colors

tqdm.pandas()


def bootstrap_mean_ci(data, ci=95):
    n_iterations = 1000
    n = len(data)
    mean_difference = np.zeros(n_iterations)
    for i in range(n_iterations):
        resample = np.random.choice(data, n, replace=True)
        mean_difference[i] = resample.mean()
    mean = mean_difference.mean()
    ci_min = np.percentile(mean_difference, (100 - ci) / 2)
    ci_max = np.percentile(mean_difference, ci + (100 - ci) / 2)
    return mean, ci_min, ci_max


df = pd.read_csv('ar_models/csv/pooled_effects/Joint RMSE for all data.csv', index_col=0)
df = df.pivot(index=['occupation_name', 'state_name'], columns=['label']).reset_index()
df.columns = ['occupation_name', 'state_name', 'employment', 'job_offer_data']
df['difference'] = ((df['employment'] / df['job_offer_data']) - 1) * 100

df['occupation_name'] = df['occupation_name'].apply(lambda occ: ' '.join([tag.capitalize() for tag in occ.split()[:6]]))
df = df.sort_values(['occupation_name', 'state_name']).reset_index(drop=True)
df_pivot = df.pivot("state_name", "occupation_name", "difference")
fig = px.imshow(df_pivot)
fig.write_html(f'try.html')

explicit_df = df.drop(['employment', 'job_offer_data'], axis=1).pivot_table(columns='occupation_name',
                                                                            index='state_name', values='difference')
df = explicit_df.reset_index().melt(["state_name"], value_name='difference')
occ_map = {
    'Assemblers': 'Assemblers',
    'Building And Related Trades Workers (excluding': 'Building Workers',
    'Business And Administration Professionals': 'Business Professionals',
    'Chief Executives, Senior Officials And Legislators': 'Chief Executives',
    'Cleaners And Helpers': 'Cleaners And Helpers',
    'Commissioned Armed Forces Officers': 'Armed Forces Officers',
    'Drivers And Mobile Plant Operators': 'Drivers And Operators',
    'Electrical And Electronics Trades Workers': 'Electrical Trades Workers',
    'Food Processing, Woodworking, Garment And Other': 'Food Processing Workers',
    'General And Keyboard Clerks': 'General Clerks',
    'Handicraft And Printing Workers': 'Handicraft Workers',
    'Health Professionals': 'Health Professionals',
    'Information And Communications Technology Professionals': 'Technology Professionals',
    'Labourers In Mining, Construction, Manufacturing And': 'Labourers',
    'Legal, Social And Cultural Professionals': 'Legal Professionals',
    'Market-oriented Skilled Agricultural Workers': 'Agricultural Workers',
    'Metal, Machinery And Related Trades Workers': 'Metal And Machinery Workers',
    'Personal Service Workers': 'Personal Service Workers',
    'Production And Specialized Services Managers': 'Production Managers',
    'Sales Workers': 'Sales Workers',
    'Science And Engineering Professionals': 'Engineering Professionals',
    'Stationary Plant And Machine Operators': 'Machine Operators',
    'Teaching Professionals': 'Teaching Professionals',
}

state_code_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv', usecols=['code', 'state'])
state_code_df = pd.concat([state_code_df, pd.DataFrame({'state': 'District of Columbia', 'code': 'DC'}, index=[0])])
state_code_df = state_code_df.set_index('state')

df['occupation_name_short'] = df['occupation_name'].apply(lambda occ: occ_map[occ])
df = df.sort_values(['state_name', 'occupation_name_short'], ascending=[True, False])
nan_color = 'black'
color_map = px.colors.get_colorscale('icefire')
mid_point = len(color_map) // 2
orig_color = color_map[mid_point][1]
color_map[mid_point] = [0.5, 'rgb(0,0,0)']
color_map.insert(mid_point, [0.5 - 1e-9, orig_color])
color_map.insert(mid_point + 2, [0.5 + 1e-9, orig_color])
color_map[0] = [0.0, '#001738']
color_map[-1] = [1.0, '#4c0815']

fig = px.density_heatmap(df,
                         "state_name",
                         "occupation_name_short",
                         "difference",
                         color_continuous_scale=color_map,
                         color_continuous_midpoint=0
                         )
subtitle_size = 60
# Customizing x-axis tick labels
xtick_font_size = 55
ytick_font_size = 55

# state_names = df['state_name'].unique()
state_names = state_code_df.loc[df['state_name'].unique()]['code']
ticktext = [f"<b>{name}</b>" for name in state_names]

fig.update_xaxes(
    title='State Name',
    tickfont=dict(size=xtick_font_size),
    titlefont=dict(size=subtitle_size),
    tickvals=np.arange(len(df['state_name'].unique())),  # Use index as tick values
    ticktext=ticktext  # Use customized tick labels
)

# Customizing y-axis tick labels
fig.update_yaxes(
    title='Occupation Name',
    tickfont=dict(size=ytick_font_size),
    titlefont=dict(size=subtitle_size),
    tickvals=np.arange(len(df['occupation_name_short'].unique())),  # Use index as tick values
    ticktext=df['occupation_name_short'].unique()  # Use occupation names as tick labels
)

# Customizing colorbar
fig.layout.coloraxis.colorbar.title = 'RMSE improvement(%)'
fig.layout.coloraxis.colorbar.orientation = 'v'
fig.layout.coloraxis.colorbar.title.side = 'right'
fig.layout.coloraxis.colorbar.tickfont.size = 45
fig.layout.coloraxis.colorbar.titlefont.size = subtitle_size

# Customizing layout
fig.layout.title.font.size = 70
fig.update_layout(title_xanchor='left', title_xref='paper')

fig.update_layout(title_xanchor='left', title_xref='paper')
pio.write_image(fig, 'rmse_heatmap.png', height=2000, width=6000)
fig.write_html(f'rmse_heatmap.html', default_height=2000, default_width=6000)
fig.write_image(f'rmse_heatmap.pdf', height=2500, width=7500)
