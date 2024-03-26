from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sentence_transformers import util
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config.config import PipeLineConfig, JobTitleConfig, device
from utils.general_utils import makedirs
from utils.job_title_utils import get_embeddings
import seaborn as sns
import plotly.express as px

embedder = PipeLineConfig().embedder
print(f'This run save to: {JobTitleConfig.folder[3:]}\nTraining using {device}')
makedirs(f"{JobTitleConfig.folder}")

input_path = JobTitleConfig.job_title_input
df = pd.read_csv(input_path, dtype=str, encoding='latin-1')
labels_map = list(df[f'{JobTitleConfig.level_prefix}_category'].unique())
num_labels = len(labels_map)
category_labels = list(df[f'{JobTitleConfig.level_prefix}_category_label'].unique())

df[f'{JobTitleConfig.level_prefix}_category'] = \
    df[f'{JobTitleConfig.level_prefix}_category'].apply(lambda cat: labels_map.index(cat))

train_df, test = train_test_split(df, train_size=JobTitleConfig.TRAIN_SIZE,
                                  stratify=df[f'{JobTitleConfig.level_prefix}_category'])

train_embeddings = get_embeddings(train_df, 'job_title', embedder).to(device)
test_embeddings = get_embeddings(test, 'job_title', embedder).to(device)

y_true = test[[f'{JobTitleConfig.level_prefix}_category']]

score_functions = [util.dot_score, util.cos_sim]
top_ks = [1, 3, 5]

for idx, (score_function, top_k) in enumerate(product(score_functions, top_ks)):
    sub_dir = f"{JobTitleConfig.folder}/score_{'dot' if score_function is util.dot_score else 'cosine'}_topk_{top_k}"
    makedirs(sub_dir)
    test_preds = util.semantic_search(test_embeddings, train_embeddings,
                                      score_function=score_function, top_k=top_k)

    preds = pd.DataFrame(
        [Counter([train_df.iloc[sub_pred['corpus_id']][f'{JobTitleConfig.level_prefix}_category'] for sub_pred in
                  pred]).most_common(1)[0][0] for pred in test_preds])

    cr = classification_report(y_true, preds, output_dict=True)
    cr = pd.DataFrame(cr).T
    cr = cr.rename({str(i): label for i, label in enumerate(category_labels)})
    cr.to_csv(f'{sub_dir}/classification_report.csv')

    cm = confusion_matrix(y_true, preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    confusion_matrix_df = pd.DataFrame(data=cm,
                                       columns=category_labels,
                                       index=category_labels)
    fig = px.imshow(confusion_matrix_df,
                    color_continuous_scale="Viridis",
                    title='prediction Heat Map'
                    )
    fig.write_html(f'{sub_dir}/heat_map.html')
    print(f"Accuracy for: {'dot' if score_function is util.dot_score else 'cosine'} score with top_k = {top_k} is "
          f'{len(test[preds.values == y_true.values]) / len(test)}')
