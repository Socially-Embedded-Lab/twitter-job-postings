import os
import sys
from glob import glob

import pandas as pd
from pandas.errors import ParserError
import pyarrow.parquet as pq

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.pipeline_utils import *
from config.config import PipeLineConfig, device, run_prefix, TASK_ID, TASK_COUNT
from utils.job_title_utils import MulticlassClassification
from sentence_transformers import util
from utils.general_utils import *

import torch
from torch.utils.data import TensorDataset, DataLoader


if __name__ == "__main__":
    print(f'This run save to: {PipeLineConfig.folder[len(run_prefix):]}')
    print(f'Will use: {torch.cuda.get_device_name(0)} as GPU')

    print(f"Running with task_id: {TASK_ID}")
    print(f"\nTotal tasks: {TASK_COUNT}")

    makedirs(f"{PipeLineConfig.folder}", exist_ok=True)
    makedirs(f"{PipeLineConfig.folder}/csv", exist_ok=True)
    makedirs(f"{PipeLineConfig.folder}/figures", exist_ok=True)
    makedirs(f"{PipeLineConfig.folder}/csv/merged", exist_ok=True)
    makedirs(f"{PipeLineConfig.folder}/csv/preds", exist_ok=True)

    if PipeLineConfig.save_intermediate_files:
        makedirs(f"{PipeLineConfig.folder}/csv/job_titles", exist_ok=True)
        makedirs(f"{PipeLineConfig.folder}/csv/location", exist_ok=True)

    embedder = PipeLineConfig().embedder
    labels_map = pd.read_csv(f'{PipeLineConfig.JOB_TITLES_DIR}/labels_map.csv')
    num_labels = len(labels_map)
    # App functionality code begins here
    print(f"Predict Named Entities with BER with {num_labels} labels!")

    try:
        mgr = LanguageResourceManager("en", PipeLineConfig.label_types, PipeLineConfig.CHK_PATH,
                                      PipeLineConfig.MODEL, PipeLineConfig.TOKENIZER, labels=PipeLineConfig.ner_tags)
    except RuntimeError:
        print("The selected checkpoint is not compatible with this BERT model.")
        print("Are you sure you have the right checkpoint?")
        exit()

    try:
        corpus_locations_embedding = torch.load(f'{PipeLineConfig.folder}/corpus_locations_embedding.tar')
    except FileNotFoundError:
        corpus_locations_embedding = embedder.encode(loc_list,
                                                     convert_to_tensor=True,
                                                     show_progress_bar=True,
                                                     normalize_embeddings=True).to(device)
        torch.save(corpus_locations_embedding, f'{PipeLineConfig.folder}/corpus_locations_embedding.tar')
    model = MulticlassClassification(num_labels)
    # push the model to GPU
    model = model.to(device)
    model.load_state_dict(
        torch.load(f'{PipeLineConfig.JOB_TITLES_DIR}/weights/saved_weights_{PipeLineConfig.JOB_TITLES_BEST_EPOCH}.pt')
    )
    model.eval()
    stages = ['Load Dataset', 'NER Tag Dataset', 'Encode Job Titles', 'Predict Job Titles',
              'Encode Location', 'Predict Locations', 'Merge DF']
    stages_dict = {idx: {'stage_name': tag, 'avg_time': 0} for idx, tag in enumerate(stages, start=1)}
    count_dict = {
        'Date': [],
        'Total Tweets': [],
        'Job Titles': [],
        'Location': [],
        'Merged': [],
    }
    try:
        dates = sorted(list({f.split('\\')[-2] for f in glob(f'{run_prefix}data/job_offer/sample/*/sample.parquet')}))
    except IndexError:
        dates = sorted(list({f.split('/')[-2] for f in glob(f'{run_prefix}data/job_offer/sample/*/sample.parquet')}))
    if not dates:
        raise Exception('There are no dates!! check glob')
    dates_split = np.array_split(dates, TASK_COUNT)
    dates = dates_split[TASK_ID - 1]
    print(f'Will predict from: {dates[0]}, until: {dates[-1]}')
    total_files = len(dates)
    time_per_stage = time.time()
    for date in tqdm(dates, desc='Dates', leave=True):
        count_dict['Date'].append(date)
        count_dict['Total Tweets'].append(0)
        count_dict['Job Titles'].append(0)
        count_dict['Location'].append(0)
        count_dict['Merged'].append(0)
        stage_idx = 1
        try:
            df = pq.read_table(f'{run_prefix}data/job_offer/sample/{date}/sample.parquet').to_pandas()
            # df = pd.read_csv(f'{run_prefix}data/job_offer/sample/{date}/sample.csv', error_bad_lines=False)
        except ParserError:
            print(f'Parsing failed for: {run_prefix}data/job_offer/sample/{date}/sample.parquet')
            continue
        except FileNotFoundError:
            print(f'No file found for: {run_prefix}data/job_offer/sample/{date}/sample.parquet')
            continue
        if df.empty:
            print(f'No lines were read for: {run_prefix}data/job_offer/sample/{date}/sample.parquet')
            continue

        count_dict['Total Tweets'][-1] = len(df)
        stages_dict[stage_idx]['avg_time'] += time.time() - time_per_stage
        time_per_stage = time.time()

        df.index = df['tweet_id']

        stage_idx = 2
        bert_preds = mgr.get_preds(df)
        if bert_preds.empty:
            print(f'no preds for: {date}')
            continue
        bert_preds.to_csv(f'{PipeLineConfig.folder}/csv/preds/{date}.csv', index=False)
        bert_preds['tag'] = bert_preds.tag.apply(lambda t: t[2:])
        tweets = bert_preds.groupby(['user_id', 'tweet_id', 'tag']).text.apply(' '.join).reset_index()
        job_titles = tweets[tweets['tag'] == 'JOB_TITLE'].reset_index(drop=True)
        job_titles['text'] = job_titles['text'].apply(lambda text: re.sub(r'\s\s+', ' ', text.replace('[SEP]', '')))
        stages_dict[stage_idx]['avg_time'] += time.time() - time_per_stage
        time_per_stage = time.time()
        count_dict['Job Titles'][-1] = len(job_titles)

        stage_idx = 3
        if job_titles.empty:
            print(f'no job titles for: {date}')
            continue
        input_embeddings = embedder.encode(job_titles.text, convert_to_tensor=True)
        batch_size = 512
        # wrap tensors
        data = TensorDataset(input_embeddings)
        # dataLoader for train set
        dataloader = DataLoader(data, batch_size=batch_size)
        stages_dict[stage_idx]['avg_time'] += time.time() - time_per_stage
        time_per_stage = time.time()

        stage_idx = 4
        total_preds = []
        for batch in dataloader:  # tqdm(dataloader, desc=stages[stage_idx - 1], leave=False)
            # push the batch to gpu
            batch = [t.to(device) for t in batch]
            model_in = batch[0]
            # deactivate autograd
            with torch.no_grad():
                # model predictions
                preds = model(model_in)
                preds = preds.max(1).indices
                preds = preds.detach().cpu().numpy()
                total_preds.extend(preds)

        res = pd.DataFrame()
        res['user_id'] = job_titles['user_id']
        res['tweet_id'] = job_titles['tweet_id']
        res['job_titles'] = job_titles['text']
        res['categories'] = total_preds
        res[['category', 'example', 'category_label']] = labels_map.loc[res['categories']].values
        if PipeLineConfig.save_intermediate_files:
            res.to_csv(f'{PipeLineConfig.folder}/csv/job_titles/{date}.csv', index=False)
        locations = tweets[tweets['tag'] == 'LOC'].reset_index(drop=True)
        locations['text'] = locations['text'].apply(
            lambda text: ' '.join(list(dict.fromkeys(text.replace('[SEP]', '').split()))))
        stages_dict[stage_idx]['avg_time'] += time.time() - time_per_stage
        time_per_stage = time.time()
        count_dict['Location'][-1] = len(locations)

        stage_idx = 5
        if locations.empty:
            print(f'no locations for: {date}')
            continue
        locations_embeddings = embedder.encode(list(locations.text),
                                               convert_to_tensor=True,
                                               normalize_embeddings=True).to(device)
        stages_dict[stage_idx]['avg_time'] += time.time() - time_per_stage
        time_per_stage = time.time()

        stage_idx = 6
        levinstein_locs = util.semantic_search(locations_embeddings, corpus_locations_embedding,
                                               score_function=util.dot_score, top_k=PipeLineConfig.top_k)
        final_locs = pd.DataFrame(
            [[loc_list[sub_loc['corpus_id']], sub_loc['score']] for loc in levinstein_locs for sub_loc in loc])
        final_locs.columns = ['user_location', 'loc_score']
        final_locs['loc_score'] = final_locs['loc_score'].apply(lambda loc_score: min(1, loc_score))
        final_locs[['address', 'lat', 'lon']] = user_locations.loc[final_locs['user_location']].reset_index()[
            ['inferred_location', 'latitude', 'longitude']]
        final_locs['original_address'] = [loc for loc in list(locations.text) for _ in range(PipeLineConfig.top_k)]
        final_locs['user_id'] = locations['user_id']
        final_locs['tweet_id'] = locations['tweet_id']

        final_locs = final_locs[['user_id', 'tweet_id', 'original_address', 'address', 'loc_score', 'lat', 'lon']]
        if PipeLineConfig.save_intermediate_files:
            final_locs.to_csv(f'{PipeLineConfig.folder}/csv/location/{date}.csv', index=False)
        stages_dict[stage_idx]['avg_time'] += time.time() - time_per_stage
        time_per_stage = time.time()

        stage_idx = 7
        merged = res.merge(final_locs, on=['user_id', 'tweet_id'], how='outer')
        merged['lat'] = merged['lat'].astype(float)
        merged['lon'] = merged['lon'].astype(float)
        merged = merged.dropna()
        if merged.empty:
            print(f'no tweets to merge for: {date}')
            continue
        merged['created_at'] = pd.to_datetime(df.loc[merged['tweet_id']].created_at.values)
        merged['input_text'] = df.loc[merged['tweet_id']].text.values
        merged = merged[['user_id', 'tweet_id', 'created_at', 'input_text', 'job_titles', 'original_address',
                         'category_label', 'categories', 'category', 'example', 'category_label', 'original_address',
                         'address', 'loc_score', 'lat', 'lon']]
        merged.to_csv(f'{PipeLineConfig.folder}/csv/merged/{date}.csv', index=False)
        stages_dict[stage_idx]['avg_time'] += time.time() - time_per_stage
        time_per_stage = time.time()

    stages_time = pd.DataFrame(stages_dict).T.set_index('stage_name') / total_files
    stages_time.to_csv(f'{PipeLineConfig.folder}/csv/execution_times_{TASK_ID}.csv')
    count_dict = pd.DataFrame(count_dict).set_index('Date')
    count_dict.to_csv(f'{PipeLineConfig.folder}/csv/count_dates_{TASK_ID}.csv')
    print(stages_time)
