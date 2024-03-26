import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from torch.optim import Adam
from transformers import BertForTokenClassification, BertTokenizer

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config.config import NERConfig, device, run_prefix, THIS_RUN
from utils.ner_utils import (
    process_df,
    load_and_prepare_data,
    get_hyperparameters,
    train_and_save_model,
)

if __name__ == "__main__":
    print(f'This run save to: {NERConfig.folder[len(run_prefix):]}/{THIS_RUN}')
    # Process and combine data
    if NERConfig.preprocess:
        process_df()

    # Create directory for storing our model checkpoints
    if not os.path.exists(NERConfig.general_ner_models_folder):
        os.mkdir(NERConfig.general_ner_models_folder)

    os.mkdir(NERConfig.ner_models_folder)

    # Create directory for storing figures for this run
    if not os.path.exists(NERConfig.general_ner_figures_folder):
        os.mkdir(NERConfig.general_ner_figures_folder)

    os.mkdir(NERConfig.ner_figures_folder)
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(NERConfig.TOKENIZER, do_lower_case=False)

    # Create dicts for mapping from labels to IDs and back
    tag2idx = {t: i for i, t in enumerate(NERConfig.label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    # Load and prepare data
    train_dataloader, valid_dataloader = load_and_prepare_data(NERConfig, tokenizer, tag2idx)
    print("Loaded training and validation data into DataLoaders.")

    # Initialize model
    model = BertForTokenClassification.from_pretrained(NERConfig.MODEL, num_labels=NERConfig.NUM_LABELS,
                                                       ignore_mismatched_sizes=True)
    model.to(device)
    print(f"Initialized model and moved it to {device}.")

    # Set hyper-parameters (optimizer, weight decay, learning rate)
    optimizer_grouped_parameters = get_hyperparameters(model, NERConfig.FULL_FINE_TUNING)
    optimizer = Adam(optimizer_grouped_parameters, lr=NERConfig.LR)
    print("Initialized optimizer and set hyper-parameters.")

    # Fine-tune model and save checkpoint every epoch
    history, cl_report = train_and_save_model(
        model,
        tokenizer,
        optimizer,
        NERConfig.EPOCHS,
        idx2tag,
        tag2idx,
        NERConfig.MAX_GRAD_NORM,
        device,
        train_dataloader,
        valid_dataloader,
        NERConfig.ner_models_folder,
        NERConfig.ner_figures_folder,
        NERConfig.DEBUG,
        show=NERConfig.SHOW
    )

    df = pd.DataFrame(history)
    df.index = df.index + 1
    history = {
        'Training accuracy': [],
        'Evaluation accuracy': [],
        'Training loss': [],
        'Evaluation loss': [],
    }

    df[['Training loss', 'Evaluation loss']].plot()
    plt.title('Training and evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim((1, NERConfig.EPOCHS))
    plt.savefig(f'{NERConfig.ner_figures_folder}/loss.jpg')
    plt.show()

    df[['Training accuracy', 'Evaluation accuracy']].plot()
    plt.title('Training and evaluation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim((1, NERConfig.EPOCHS))
    plt.savefig(f'{NERConfig.ner_figures_folder}/accuracy.jpg')
    plt.show()

    df.to_csv(f"{NERConfig.ner_models_folder}/history.csv", index=False)
    cl_report.to_csv(f"{NERConfig.ner_models_folder}/cl_report.csv", index=False)
    cl_report.to_csv(f"{NERConfig.ner_models_folder}/cl_report.csv", index=False)
    fig = px.bar(cl_report, x='index', y=cl_report.columns, animation_frame='epoch', barmode='group')
    fig.write_html(f"{NERConfig.ner_figures_folder}/cl_report.html")
