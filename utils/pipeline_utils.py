# Imports
import os
import re
import sys
import time
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from fuzzywuzzy import process
from geopy.exc import GeocoderTimedOut
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertForTokenClassification
from typing import List
import plotly.express as px

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config.config import run_prefix, device, PipeLineConfig, NERConfig

tqdm.pandas()

user_locations = pd.read_csv(f'{run_prefix}data/csv/us_cities.csv', dtype=str)
loc_list = list(user_locations['user_location'])
user_locations = user_locations.set_index('user_location')


class LanguageResourceManager:
    """
    Manages resources for each language, such as the models. Also acts as a
    convenient interface for getting predictions.
    """

    def __init__(self, lang, label_types, chk_path,
                 model_name='bert-base-cased', tokenizer_name='bert-base-cased',
                 labels: List[str] = PipeLineConfig.ner_tags):

        self.lang = lang
        self.label_types = label_types
        self.label_dict = {i: t for i, t in enumerate(self.label_types)}
        self.num_labels = len(self.label_types)
        self.chk_path = chk_path
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.labels = labels

    def get_preds(self, df: pd.DataFrame):
        return self.get_bert_pred_df(df)

    def get_bert_pref_text(self, input_text):
        encoded_text = self.tokenizer.encode(input_text)
        word_pieces = [self.tokenizer.decode(tok).replace(' ', '') for tok in encoded_text]

        input_ids = torch.tensor(encoded_text).unsqueeze(0).long().to(device)
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0).long().to(device)
        outputs = self.model(input_ids, labels=labels)
        _, scores = outputs[:2]
        scores = scores.detach().cpu().numpy()

        label_ids = np.argmax(scores, axis=2)
        preds = [self.label_dict[i] for i in label_ids[0]]

        wp_preds = list(zip(word_pieces, preds))
        toplevel_preds = [pair[1] for pair in wp_preds if '##' not in pair[0]]
        str_rep = ' '.join([t[0] for t in wp_preds]).replace(' ##', '').split()
        return toplevel_preds, str_rep

    def get_bert_pref_texts(self, input_texts):
        encoded_texts = [self.tokenizer.encode(input_text) for input_text in input_texts]
        word_pieces = [[self.tokenizer.decode(tok).replace(' ', '') for tok in encoded_text] for
                       encoded_text in encoded_texts]
        input_ids = torch.tensor(pad_sequences(
            encoded_texts,
            maxlen=NERConfig.MAX_LEN,
            dtype="long",
            truncating="post",
            padding="post",
        )).long().to(device)
        labels = torch.tensor([1] * input_ids.size(0) * input_ids.size(1)).unsqueeze(0).long().to(device)
        outputs = self.model(input_ids, labels=labels)
        _, scores = outputs[:2]
        scores = scores.detach().cpu().numpy()

        label_ids = np.argmax(scores, axis=2)
        preds = [[self.label_dict[i] for i in label_id] for label_id in label_ids]
        toplevel_preds = np.array([[tok for (tok, piece) in zip(pred, wp) if '##' not in piece] for wp, pred in
                                   zip(word_pieces, preds)], dtype='object')
        str_rep = np.array([' '.join(wp[:NERConfig.MAX_LEN]).replace(' ##', '').split() for wp in word_pieces],
                           dtype='object')
        return str_rep, toplevel_preds

    def pred_line(self, tweet_id, input_text):
        toplevel_preds, str_rep = self.get_bert_pref_text(input_text)

        # If resulting string length is correct, create prediction columns for each tag
        if len(str_rep) == len(toplevel_preds):
            preds_final = list(zip(str_rep, toplevel_preds))
            b_preds_df = pd.DataFrame(preds_final)
            b_preds_df.columns = ['text', 'tag']
            for tag in self.label_types[:-1]:
                b_preds_df[f'b_pred_{tag.lower()}'] = np.where(
                    b_preds_df['tag'].str.contains(tag), 1, 0
                )
            b_preds_df['tweet_id'] = tweet_id
            b_preds_df = b_preds_df[b_preds_df['tag'].isin(self.labels)]
            return b_preds_df.loc[:, 'text':][['tweet_id', 'text', 'tag']]
        else:
            print('Could not match up output string with preds.')
            exit()

    def pred_lines(self, user_ids, tweet_ids, input_texts):
        str_rep, toplevel_preds = self.get_bert_pref_texts(input_texts)
        # If resulting string length is correct, create prediction columns for each tag
        matching = [len(wp) == len(tp) for wp, tp in zip(str_rep, toplevel_preds)]
        if not all(matching):
            print('Could not match up output string with preds.')
            str_rep, toplevel_preds, user_ids, tweet_ids = str_rep[matching], toplevel_preds[matching], user_ids[matching], tweet_ids[matching]
        preds_final = list(zip(str_rep, toplevel_preds, user_ids, tweet_ids))
        b_preds_df = pd.DataFrame(preds_final)
        b_preds_df.columns = ['text', 'tag', 'user_id', 'tweet_id']
        b_preds_df = b_preds_df.explode(['text', 'tag'])
        for tag in self.label_types[:-1]:
            b_preds_df[f'b_pred_{tag.lower()}'] = np.where(
                b_preds_df['tag'].str.contains(tag), 1, 0
            )
        b_preds_df = b_preds_df[b_preds_df['tag'].isin(self.labels)]
        return b_preds_df.loc[:, 'text':][['user_id', 'tweet_id', 'text', 'tag']]

    def get_bert_pred_df(self, src_df: pd.DataFrame, single_line=False):
        """
        Uses the model to make a prediction, with batch size 1.
        """
        if single_line:
            return pd.concat(list(map(self.pred_line, src_df['tweet_id'], src_df['text'])))
        preds = []
        splits = math.ceil(len(src_df) / NERConfig.BATCH_SIZE)
        for user_ids, tweet_ids, input_texts in zip(  # tqdm(
                np.array_split(src_df['user_id'], splits),
                np.array_split(src_df['tweet_id'], splits),
                np.array_split(src_df['text'], splits)):  # ,desc='NER Tag Dataset', leave=False, total=splits)
            preds.append(self.pred_lines(user_ids, tweet_ids, input_texts))
        return pd.concat(preds)

    def load_model_and_tokenizer(self, state_dict_name='model_state_dict'):
        """
        Loads model from a specified checkpoint path. Replace `CHK_PATH` at the top of
        the script with where you are keeping the saved checkpoint.
        (Checkpoint must include a model state_dict, which by default is specified as
        'model_state_dict,' as it is in the `main.py` script.)
        """
        if self.model_name is None:
            self.model_name = 'DeepPavlov/bert-base-cased-conversational'

        checkpoint = torch.load(self.chk_path)
        model_state_dict = checkpoint[state_dict_name]
        model = BertForTokenClassification.from_pretrained(
            self.model_name, num_labels=model_state_dict['classifier.weight'].shape[0]
        )
        model.load_state_dict(model_state_dict)
        model.to(device)
        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name, do_lower_case=False)

        return model, tokenizer


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)


def get_location(geo_locator, loc: str):
    try:
        time.sleep(1)
        if loc.endswith('US'):
            loc = loc[:-2]
        location = geo_locator.geocode(loc)
        if location is None or 'United States' not in location.address:
            location = geo_locator.geocode(loc + ', USA')
            if location is None or 'United States' not in location.address:
                location = geo_locator.geocode(loc.split(' ')[-1] + ' ,USA')
        return location
    except GeocoderTimedOut:
        print(loc)
        return None


def get_loc_by_levinstein(loc: str):
    t = process.extractOne(loc, loc_list)
    return t[0], t[1]


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        text = text.replace(prefix, '', 1)
    return text


def clean(text):
    # Remove all urls from the tweet
    clean_text = re.sub(r'http\S+t.\S+', '', text)
    # Remove Hashtags
    clean_text = re.sub('#', '', clean_text)
    # remove US as it gives no additional input
    clean_text = re.sub('US', ' ', clean_text)
    clean_text = re.sub(r'[mM]gmt', 'management', clean_text)
    clean_text = re.sub(r'[mM]gr', 'manager', clean_text)
    # remove all white space
    clean_text = re.sub(r'\s\s+', ' ', clean_text)
    # trim
    clean_text = clean_text.strip()
    # If some tweets were left empty then remove them
    return clean_text


# Define formula applying that to a dataframe
def return_levinstein_location(data):
    return pd.Series(data.progress_apply(get_loc_by_levinstein))


# Define formula to parallelize the code on multiple cores
def parallelize_dataframe(df, func, n_cores=8):
    rdf = []
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    rdf.extend(pool.map(func, df_split))
    pool.close()
    pool.join()
    return pd.concat(rdf)


def get_data_from_stats(df, data_label):
    return_df = df[['year', 'month', 'state_name', 'occupation_name', data_label]]
    return_df = return_df.rename({data_label: 'value'}, axis=1)
    return return_df


def get_date_from_df(df):
    return df['year'].map(str) + '-' + df['month'].map(str).apply(
        lambda m: m.zfill(2)) + '-01'


def plot_and_save_line(df, sort_values, x, y, color, title, labels, name, symbol=None):
    fig = px.line(df.dropna().sort_values(by=sort_values), x=x, y=y,
                  color=color,
                  markers=True, template="plotly_white",
                  title=title, animation_frame='state_name',
                  labels=labels, symbol=symbol)
    fig.write_html(f'{PipeLineConfig.folder}/figures/{name}.html')
