from collections import defaultdict, Counter

import pyarrow.parquet as pq
import pandas as pd
from glob import glob
from tqdm import tqdm
import plotly.express as px
import os
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import re
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from utils.general_utils import makedirs


def clean(text):
    try:
        # Remove all urls from the tweet
        clean_text = re.sub(r'http\S+', '', text)
        clean_text = re.sub(r'www\S+', '', clean_text)
        clean_text = re.sub(r'RT\S+', '', clean_text)
        clean_text = ' '.join(re.split(r'(?<![A-Z\W])(?=[A-Z])', clean_text))
        # Remove Hashtags
        clean_text = re.sub('#', '', clean_text)
        # remove US as it gives no additional input
        clean_text = re.sub('US', ' ', clean_text)
        clean_text = re.sub(r"[^a-zA-Z_.,:-\\']", ' ', clean_text)
        clean_text = ' '.join(
            ' '.join(sa) for sa in re.findall('(?:(\@[0-9a-zA-Z_]+))|([^A-Z]*[A-Z]*[^A-Z]*)', clean_text))
        # remove all white space
        clean_text = re.sub(r'\s\s+', ' ', clean_text)
        # trim
        clean_text = clean_text.strip()
        # If some tweets were left empty then remove them
    except TypeError:
        clean_text = ''
    return clean_text


run_prefix = '' if os.getcwd().endswith('Labour-Market-Statistics') else '../'

cpu_device = torch.device('cpu')
WITH_EMBEDDINGS = False
DELETE_FILES = False

try:
    dates = sorted(list({f.split('\\')[-2] for f in glob(f'{run_prefix}job_offer/sample/*/*.parquet')}))
except IndexError:
    dates = sorted(list({f.split('/')[-2] for f in glob(f'{run_prefix}job_offer/sample/*/*.parquet')}))
if not dates:
    raise Exception('There are no dates!! check glob')

for date in dates:
    makedirs(f'{run_prefix}job_offer/sample/{date}/duplicates', exist_ok=True)

tweet_id_to_user_id = pd.DataFrame(columns=['user_id', 'tweet_id'])
dedup_count = defaultdict(list)
user_id_counts = Counter()

tweet_id = 0
if WITH_EMBEDDINGS:
    embedder = SentenceTransformer('all-mpnet-base-v2')
for date in tqdm(dates):
    files = glob(f'{run_prefix}job_offer/sample/{date}/*_*.parquet')
    dfs = []
    for f in files:
        try:
            dfs.append(pq.read_table(f).to_pandas())
        except pd.errors.ParserError as pe:
            try:
                dfs.append(pd.read_parquet(f, on_bad_lines='skip').to_pandas())
            except Exception as e:
                print(f'Bad file: {f}, Error: {pe}')
    if not dfs:
        continue
    dedup_count['date'].append(date)
    df = pd.concat(dfs).reset_index(drop=True)
    df['tweet_id'] = list(range(tweet_id, tweet_id + len(df)))
    tweet_id += len(df)
    total_tweets = len(df)
    dedup_count['initial'].append(total_tweets)

    df['clean_text'] = pd.Series(np.vectorize(clean)(df.text)).replace('', np.nan)
    df_clean = df[df.isna()['clean_text']]
    if not df_clean.empty:
        df_clean.to_csv(f'{run_prefix}job_offer/sample/{date}/duplicates/clean_drop_na.csv', index=False)
    df = df.dropna(subset=['clean_text'])
    clean_len = len(df)
    dedup_count['clean_removed'].append(total_tweets - clean_len)
    df_dup = df[df.duplicated(subset=['clean_text', 'user_id'])]
    if not df_dup.empty:
        df_dup.to_csv(f'{run_prefix}job_offer/sample/{date}/duplicates/same_text.csv', index=False)
    df = df.drop_duplicates(subset=['clean_text', 'user_id'])
    same_text_len = len(df)
    dedup_count['same_text'].append(clean_len - same_text_len)
    df_links = df.explode('links')
    df_links = df_links[~(df_links['links'].astype(str) == 'nan')]
    df_links = df_links[df_links.duplicated('links')]
    duplicated_ids = df_links.tweet_id
    df_links = df[df['tweet_id'].isin(duplicated_ids)]
    if not df_links.empty:
        df_links.to_csv(f'{run_prefix}job_offer/sample/{date}/duplicates/same_links.csv', index=False)
    df = df[~df['tweet_id'].isin(duplicated_ids)]
    links_len = len(df)
    dedup_count['same_link'].append(same_text_len - links_len)
    # embed all
    if WITH_EMBEDDINGS:
        same_month_embedding = embedder.encode(list(df['clean_text']),
                                               convert_to_tensor=True,
                                               show_progress_bar=True,
                                               normalize_embeddings=True).to(cpu_device)
        communities = util.community_detection(same_month_embedding, threshold=0.95, min_community_size=1)
        unique_ids = [community[0] for community in communities]
        for idx, community in enumerate(communities):
            if len(community) > 10:
                df.iloc[community].to_csv(f'{run_prefix}job_offer/sample/{date}/duplicates/{idx}.csv', index=False)
        df = df.iloc[unique_ids]
        dedup_count['duplicates'].append(links_len - len(df))
    df['text'] = df.clean_text
    df = df.drop('clean_text', axis=1)
    df.to_parquet(f'{run_prefix}job_offer/sample/{date}/sample.parquet', index=False)
    dedup_count['total_tweets'].append(len(df))
    user_id_counts.update(df['user_id'].astype(str))
    tweet_id_to_user_id = pd.concat([tweet_id_to_user_id, df[['user_id', 'tweet_id']]], ignore_index=True)
    if DELETE_FILES and (len(dfs) == len(files)):
        for file in files:
            os.remove(file)

dedup_count = pd.DataFrame(dedup_count)
dedup_count = dedup_count.set_index('date')
dedup_count['Duplicates Percent'] = \
    dedup_count.drop(['initial', 'total_tweets'], axis=1).sum(axis=1) / dedup_count['initial'] * 100
dedup_count.to_csv(f'{run_prefix}job_offer/dedup_counts.csv')
fig = px.bar(dedup_count[['Duplicates Percent']],
             title='Percent of Duplicates per month',
             barmode='group')
fig.update_layout(showlegend=False)
fig.write_html(f'{run_prefix}job_offer/dedup_counts.html')
print(dedup_count[['initial', 'total_tweets']])
print(dedup_count[['initial', 'total_tweets']].sum())

user_id_counts = pd.DataFrame(user_id_counts.items(), columns=['User ID', 'Number of Tweets'])\
    .set_index('User ID').sort_values('Number of Tweets', ascending=False)
print('Saving Dupes')
user_id_counts.head(100).to_csv(f'{run_prefix}job_offer/user_counts_dedup.csv')
print('Saved Dupes')
tweet_id_to_user_id = tweet_id_to_user_id.set_index('tweet_id')
print('Saving Ids')
tweet_id_to_user_id.to_csv(f'{run_prefix}job_offer/tweet_id_to_user_id.csv')
print('Saved Ids')
