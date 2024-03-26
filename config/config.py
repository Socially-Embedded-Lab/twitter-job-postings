import os
import datetime as dt

import torch
# Specify device
from sentence_transformers import SentenceTransformer


def get_env_var(varname, default):
    if os.environ.get(varname) is not None:
        var = int(os.environ.get(varname))
    else:
        var = default
    return var


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THIS_RUN = dt.datetime.now().strftime('%d.%m.%Y, %H.%M.%S')
model_names = {'pavlov': 'DeepPavlov/bert-base-cased-conversational',
               'bert': 'bert-base-cased',
               'job_bert': 'jjzha/jobbert-base-cased',
               'bert_ner': 'dslim/bert-base-NER'
               }
run_prefix = '' if os.getcwd().endswith('Labour-Market-Statistics') else '../'
data_folder = f'{run_prefix}data'
base_labels = ['JOB_TITLE', 'LOC', 'MISC', 'ORG']
IOB = False
TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 1)
TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
sample_freq = 'M'


class ConvertConfig:
    LABELS = ['ORG', 'LOC', 'JOB_TITLE', 'MISC']

    job_offer_file = f'{data_folder}/csv/US_ner_clean.csv'
    job_offer_file_train = f'{data_folder}/csv/train.csv'
    job_offer_file_test = f'{data_folder}/csv/test.csv'
    ner_train_output = f'{data_folder}/tsv/ner_train_output.txt'
    ner_test_output = f'{data_folder}/tsv/ner_test_output.txt'
    iob = IOB


class SpacyConfig:
    LABELS = ['ORG', 'LOC', 'JOB_TITLE', 'MISC']

    spacy_train_output = 'train_set_bert.tsv'
    spacy_test_output = 'test_set_bert.tsv'
    ner_train_output = 'ner_train_output.txt'
    ner_test_output = 'ner_test_output.txt'


class NERConfig:
    folder = f'{run_prefix}ner_model'
    label_types = [f'{prefix}-{label}' for label in base_labels for prefix in (['B', 'I'] if IOB else ['B'])] + ['O']
    model_type = 'pavlov'
    language = 'en'
    iob = IOB
    train_data_path = f'{data_folder}/tsv/ner_train_output.txt'
    dev_data_path = f'{data_folder}/tsv/ner_test_output.txt'
    test_data_path = f'{data_folder}/tsv/ner_test_output.txt'
    general_ner_models_folder = f'{folder}/models'
    ner_models_folder = f'{general_ner_models_folder}/{THIS_RUN}'
    general_ner_figures_folder = f'{folder}/figs'
    ner_figures_folder = f'{general_ner_figures_folder}/{THIS_RUN}'
    MAX_LEN = 60
    BATCH_SIZE = 32
    EPOCHS = 10
    MODEL = model_names[model_type]
    TOKENIZER = model_names[model_type]
    MAX_GRAD_NORM = 1.0
    LR = 1e-4
    NUM_LABELS = len(label_types)
    FULL_FINE_TUNING = True
    preprocess = True
    DEBUG = True
    SHOW = True


class JobTitleConfig:
    level_prefix = 'sub_major'
    merge = True and (level_prefix != 'major')
    folder = f"{run_prefix}job_title_classifier/{level_prefix}/{THIS_RUN}{'_merged' if merge else ''}"
    embedder_name = 'all-mpnet-base-v2'
    job_title_input = f'{data_folder}/csv/job_titles.csv'
    category_mapping = f'{data_folder}/csv/category_mapping.csv'
    DEBUG = False
    DEBUG_STEPS = 50
    TRAIN_SIZE = 0.55
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LR = 1e-4
    LRS = [1e-4]#[1e-2, 1e-3, 1e-4]
    DR = 0.2
    DRS = [0.2]#, 0.3, 0.4, 0.5]
    EARLY_STOPPING = True
    EARLY_STOPPING_THRESHOLD = 20
    FULL_FINE_TUNING = False
    REGULARIZATION = False
    regularization_type = "l2"
    regularization_types = ['', "l2"]
    encoding_length = 768
    macro_recall = ['recall_at_1_macro', 'recall_at_5_macro']
    micro_recall = ['recall_at_1_micro', 'recall_at_5_micro']
    callback_metrics = ['accuracy_micro', 'accuracy_macro'] + macro_recall + micro_recall
    additional_metrics = ['losses', 'mrr']

    @property
    def embedder(self):
        return SentenceTransformer(self.embedder_name)


class PipeLineConfig:
    merge = JobTitleConfig.merge
    folder = f"{run_prefix}pipeline/{JobTitleConfig.level_prefix}/{'merged' if JobTitleConfig.merge else 'not_merged'}"
    JOB_TITLES_DIR, JOB_TITLES_BEST_EPOCH = ('07.01.2023, 09.36.06', 38) if JobTitleConfig.level_prefix == 'major'\
        else ('12.05.2022, 18.46.05_merged', 49) if JobTitleConfig.merge else ('12.05.2022, 18.53.09', 67)
    JOB_TITLES_DIR = f'{run_prefix}job_title_classifier/{JobTitleConfig.level_prefix}/{JOB_TITLES_DIR}'
    NER_MODEL_DIR = '29.07.2023, 11.28.05' if IOB else '12.05.2022, 17.54.22'
    NER_MODEL_EPOCH = 6 if IOB else 1
    CHK_PATH = f'{run_prefix}ner_model/models/{NER_MODEL_DIR}/train_checkpoint_epoch_{NER_MODEL_EPOCH}.tar'
    PLOT = False
    embedder_name = 'all-mpnet-base-v2'
    input_file = '13M_sample'
    label_types = NERConfig.label_types
    ner_tags = [s for s in NERConfig.label_types if s.endswith(('LOC', 'JOB_TITLE'))]
    model_type = NERConfig.model_type
    MODEL = model_names[model_type]
    TOKENIZER = model_names[model_type]
    top_k = 1
    save_intermediate_files = True

    @property
    def embedder(self):
        return SentenceTransformer(self.embedder_name)


class ARConfig:
    DEBUG = False
    PLOT = True
    LOG = False
    LOG_FACTOR = 5e-1
    FIXED_EFFECT = True

    TARGET_VAR = 'employment'
    PREFIX = '_rate'
