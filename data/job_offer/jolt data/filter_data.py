import pandas as pd


def filter_df(df, dataelement_code):
    df = df.dropna().query("survey_abbreviation=='JT'")
    df = df.query("seasonal_code=='S'")
    df = df.query("industry_code==0")
    df = df.query("area_code==0")
    df = df.query("sizeclass_code==0")
    df = df.query("dataelement_code==@dataelement_code")
    df = df.query("ratelevel_code=='L         '")
    df = df.drop(
        ['survey_abbreviation', 'seasonal_code', 'industry_code', 'area_code', 'sizeclass_code', 'dataelement_code',
         'ratelevel_code'], axis=1)
    df['state'] = df['state_code'].apply(lambda state_code: state_map.loc[state_code].state_text)
    df = df[['state_code', 'state', 'year', 'month', '       value']]
    df.columns = ['state_code', 'state', 'year', 'month', 'value']
    df = df.set_index(['state_code', 'state', 'year', 'month'])
    return df


sheet_to_df_map = pd.read_excel('data/job_offer/jolt data/jolt_new.xlsx', sheet_name=None)

state_map = sheet_to_df_map['jolt state'].set_index('state_code')
openings = filter_df(sheet_to_df_map['jolt openings'], 'JO')
separation = filter_df(sheet_to_df_map['jolt separation'], 'TS')
hires = filter_df(sheet_to_df_map['jolt hires'], 'HI')


with pd.ExcelWriter('data/job_offer/jolt data/jolt_new.xlsx') as writer:
    state_map.to_excel(writer, sheet_name='jolt state')
    openings.to_excel(writer, sheet_name='jolt openings', index=False)
    separation.to_excel(writer, sheet_name='jolt separation', index=False)
    hires.to_excel(writer, sheet_name='jolt hires', index=False)

merged = openings.copy().rename({'value': 'JOLTS Job openings'}, axis=1)
merged['JOLTS Hires'] = hires.value
merged['JOLTS Separations'] = separation.value

merged.to_csv('data/job_offer/jolt data/jolt_data.csv')