from collections import defaultdict, Counter

import pandas as pd
from glob import glob
from tqdm import tqdm
import plotly.express as px
import os
import pyarrow.parquet as pq
import sys
import shutil
from datetime import datetime

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from config.config import sample_freq
from utils.general_utils import makedirs


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


run_prefix = '' if os.getcwd().endswith('Labour-Market-Statistics') else '../'
files = glob(f'{run_prefix}job_offer/parquet2/*.parquet')
if os.path.exists(f'{run_prefix}job_offer/sample/'):
    shutil.rmtree(f'{run_prefix}job_offer/sample/')
MULTIPLIER = 1

counts = defaultdict(int)
user_id_counts = Counter()
for idx, f in tqdm(enumerate(files), total=len(files)):
    df = pq.read_table(f).to_pandas().rename(
        {'tweet_timestamp': 'created_at', 'tweet_text': 'text', 'tweet_urls': 'links', 'job_offer': 'job_offer_score'}, axis=1)
    df['created_at'] = df['created_at'].apply(lambda x: datetime.fromtimestamp(x))
    df = df[~df.text.isna()]
    df_sample = df.sample(frac=MULTIPLIER)

    df_sample = df_sample[df_sample['text'].map(lambda l: len(l) > 0)].reset_index(drop=True)

    for date, resample in df_sample.resample(sample_freq, on='created_at'):
        if resample.empty:
            continue
        makedirs(f'{run_prefix}job_offer/sample/{date.date()}', exist_ok=True)
        resample.to_parquet(f'{run_prefix}job_offer/sample/{date.date()}/sample_{idx}.parquet', index=False)
        counts[date] += len(resample)
        user_id_counts.update(resample['user_id'].astype(str))


counts = pd.DataFrame(counts.items(), columns=['Date', 'Number of Tweets']).set_index('Date')
fig = px.line(counts.dropna())
fig.write_html(f'{run_prefix}job_offer/dist.html')
counts.to_csv(f'{run_prefix}job_offer/dist.csv')
print(f'Total tweets: {counts.sum()}')

user_id_counts = pd.DataFrame(user_id_counts.items(), columns=['User ID', 'Number of Tweets'])\
    .set_index('User ID').sort_values('Number of Tweets', ascending=False)
print(f"Total number of users: {len(user_id_counts)}")
print('Saving All counts')
user_id_counts.head(100).to_csv(f'{run_prefix}job_offer/user_counts_total.csv')
print('Saved All counts')
