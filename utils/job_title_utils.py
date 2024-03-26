import os
import sys
from itertools import product
from typing import List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import Recall, Accuracy
import time

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config.config import JobTitleConfig, device
from utils.general_utils import *

tqdm.pandas()


def load_data(column_name='job_title', input_path=None, merge=False):
    if input_path is None:
        input_path = JobTitleConfig.job_title_input
    df = pd.read_csv(input_path, dtype=str, encoding='latin-1')
    df = df[~df.duplicated(subset=[column_name])].reset_index(drop=True)
    if merge:
        category_mapping = pd.read_csv(JobTitleConfig.category_mapping, index_col=0)
        df[f'{JobTitleConfig.level_prefix}_category'] = \
            category_mapping.loc[df[f'{JobTitleConfig.level_prefix}_category'].astype(int)].values
    df[f'{JobTitleConfig.level_prefix}_category'].value_counts().to_csv(f'{JobTitleConfig.folder}/value_count.csv')
    return df


def get_embeddings(df: pd.DataFrame, column_name, embedder):
    df_input = list(df[column_name])
    return embedder.encode(df_input, convert_to_tensor=True, normalize_embeddings=True)


def get_labels(df: pd.DataFrame, column_name):
    return torch.tensor(list(df[column_name]))


class ModelArchLayered(nn.Module):
    def __init__(self, num_of_labels,
                 dropout_rate=JobTitleConfig.DR, input_size=JobTitleConfig.encoding_length, dims: List = None):
        super(ModelArchLayered, self).__init__()
        self.name = 'Layered'
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.num_of_labels = num_of_labels
        if dims is None:
            self.dims = [512, 256, 128]
        else:
            self.dims = dims
        self.input_layer = nn.Linear(self.input_size, self.dims[0])
        self.fc_layers = nn.ModuleList()
        self.num_of_hidden_layers = len(self.dims) - 1
        for input_layer in range(self.num_of_hidden_layers):
            self.fc_layers.append(nn.Linear(self.dims[input_layer], self.dims[input_layer + 1]))
        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        # relu activation function
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.dims[-1], self.num_of_labels)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, input_tensor: torch.Tensor):
        # pass the inputs to the model
        x = self.input_layer(input_tensor)
        for layer in self.fc_layers:
            x = layer(x)
            x = self.dropout(x)
            x = self.relu(x)
        x = self.output_layer(x)
        # apply softmax activation
        x = self.softmax(x)
        return x


class MulticlassClassification(nn.Module):
    def __init__(self, num_of_labels, dropout_rate=JobTitleConfig.DR, num_feature=JobTitleConfig.encoding_length):
        super(MulticlassClassification, self).__init__()
        self.name = 'Multiclass Classification'
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_of_labels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class MulticlassClassificationConv(nn.Module):
    def __init__(self, num_of_labels, num_feature=JobTitleConfig.encoding_length):
        super(MulticlassClassificationConv, self).__init__()
        self.name = 'Multiclass Classification Conv'
        self.conv1 = nn.Conv1d(num_feature, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.layer_out = nn.Linear(64, num_of_labels)
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(1)

    def forward(self, x):
        x = x.reshape(*x.size(), 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = x.squeeze(2)

        x = self.layer_out(x)

        return x


def flat_accuracy(valid_tags, pred_tags):
    """
    Define a flat accuracy metric to use while training the model.
    """

    return (np.array(valid_tags) == np.array(pred_tags)).mean()


def get_mrr(indices: torch.Tensor, targets: torch.Tensor):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = torch.nonzero(targets == indices)
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr.item()


def get_callbacks(num_classes):
    callbacks = {}
    for callback in JobTitleConfig.callback_metrics:
        args = callback.split('_')
        if len(args) == 2:
            callbacks[callback] = globals()[args[0].capitalize()](num_classes=num_classes, average=args[1])
        elif len(args) == 4:
            if int(args[2]) < num_classes:
                callbacks[callback] = globals()[args[0].capitalize()](num_classes=num_classes, top_k=int(args[2]),
                                                                      average=args[3])
        else:
            raise Exception('Problem in get_macros')
    return callbacks


def init_history():
    history = defaultdict(list)
    for history_label, callback in product(['train', 'valid'],
                                           JobTitleConfig.callback_metrics + JobTitleConfig.additional_metrics):
        history[f'{history_label}_{callback}'] = []
    return history


def get_loss_function(model, regularization, regularization_type, loss_func):
    if not regularization:
        return loss_func

    def loss_fn(output, target):
        reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            if regularization_type == 'l1':
                reg_hyper_param = 0.1
                # Compute the L1 regularization term
                reg = reg + torch.sum(torch.abs(param))
            else:
                reg_hyper_param = 0.1
                # Compute the L2 regularization term
                reg = reg + torch.sum(param ** 2)

        # Add the L1 regularization term to the loss
        return loss_func(output, target) + reg_hyper_param * reg

    return loss_fn


# function to train the model
def train(data_loader, model, optimizer, loss_function, num_labels, callbacks, history, history_label='train'):
    model.train()
    total_loss = 0
    # total_accuracy = 0
    total_mrr = 0
    for callback_name in callbacks.keys():
        locals()[f'total_{callback_name}'] = 0
    # empty list to save model predictions
    total_preds = []
    # iterate over batches
    for step, batch in enumerate(data_loader):
        # progress update after every 50 batches.
        if JobTitleConfig.DEBUG and step % JobTitleConfig.DEBUG_STEPS == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(data_loader)))
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        model_in, labels = batch
        # clear previously calculated gradients
        optimizer.zero_grad()
        # get model predictions for the current batch
        preds = model(model_in)
        # compute the loss between actual and predicted values
        loss = loss_function(preds, labels)
        # add on to the total loss
        total_loss += loss.item()
        # get the indices of predicted labels
        indices = preds.max(1).indices.detach().cpu()
        # transform from cuda to CPU
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        # calculate accuracy
        # total_accuracy += flat_accuracy(indices, labels)
        # calculate MRR
        total_mrr += get_mrr(torch.topk(preds, num_labels)[1], labels)
        # get call backs metrics
        for callback_name, callback in callbacks.items():
            locals()[f'total_{callback_name}'] += callback(preds, labels).item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # model predictions are stored on GPU. So, push it to CPU
        preds = indices.numpy()
        # append the model predictions
        total_preds.append(preds)
    # compute the training loss of the epoch
    avg_loss = total_loss / len(data_loader)
    # avg_accuracy = total_accuracy / len(data_loader)
    avg_mrr = total_mrr / len(data_loader)
    for callback_name in callbacks.keys():
        locals()[f'avg_{callback_name}'] = locals()[f'total_{callback_name}'] / len(data_loader)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    # returns the loss and predictions
    history[f'{history_label}_losses'].append(avg_loss)
    # history[f'{history_label}_accuracy'].append(avg_accuracy)
    history[f'{history_label}_mrr'].append(avg_mrr)
    for callback_name in callbacks.keys():
        history[f'{history_label}_{callback_name}'].append(locals()[f'avg_{callback_name}'])
    return avg_loss, history[f'{history_label}_accuracy_micro'][-1], total_preds


# function for evaluating the model
def evaluate(data_loader, model, loss_function, num_labels, callbacks, history, history_label='valid'):
    if JobTitleConfig.DEBUG:
        print("\nEvaluating...")
    # deactivate dropout layers
    model.eval()
    total_loss = 0
    # total_accuracy = 0
    total_mrr = 0
    for callback_name in callbacks.keys():
        locals()[f'total_{callback_name}'] = 0
    # empty list to save the model predictions
    total_preds = []
    # iterate over batches
    for step, batch in enumerate(data_loader):
        # Progress update every 50 batches.
        if JobTitleConfig.DEBUG and step % JobTitleConfig.DEBUG_STEPS == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(data_loader)))
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        model_in, labels = batch
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(model_in)
            # compute the validation loss between actual and predicted values
            loss = loss_function(preds, labels)
            # add on to the total loss
            total_loss += loss.item()
            # get the indices of predicted labels
            indices = preds.max(1).indices.detach().cpu()
            # transform from cuda to CPU
            preds = preds.detach().cpu()
            labels = labels.detach().cpu()
            # calculate MRR
            total_mrr += get_mrr(torch.topk(preds, num_labels)[1], labels)
            # get call backs metrics
            for callback_name, callback in callbacks.items():
                locals()[f'total_{callback_name}'] += callback(preds, labels).item()
            preds = indices.numpy()
            total_preds.append(preds)
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(data_loader)
    # avg_accuracy = total_accuracy / len(data_loader)
    avg_mrr = total_mrr / len(data_loader)
    for callback_name in callbacks.keys():
        locals()[f'avg_{callback_name}'] = locals()[f'total_{callback_name}'] / len(data_loader)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    # Calculate elapsed time in minutes.
    history[f'{history_label}_losses'].append(avg_loss)
    # history[f'{history_label}_accuracy'].append(avg_accuracy)
    history[f'{history_label}_mrr'].append(avg_mrr)
    for callback_name in callbacks.keys():
        history[f'{history_label}_{callback_name}'].append(locals()[f'avg_{callback_name}'])
    return avg_loss, history[f'{history_label}_accuracy_micro'][-1], total_preds


def plot_stats(df, columns, title, x_label, y_label, save_postfix, best_epoch, show=False, x_lim=None, v_lim=1,
               save_dir=None):
    columns = [column for column in columns if column in df.columns]
    ax = df[columns].plot()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_lim is not None:
        plt.xlim(*x_lim)
    ax.vlines(best_epoch, 0, v_lim, linestyles='dashed')
    ax.annotate(f'Best Epoch - {best_epoch}', xy=(best_epoch, 0.3), rotation=90, va='bottom')
    best_row = df.loc[best_epoch]
    ax.legend(
        labels=["{} (best={:.3f})".format(column, best_row[column]) for column in columns],
        title='Best Epoch',
        loc='lower right')
    plt.savefig(f'{JobTitleConfig.folder if save_dir is None else save_dir}/{save_postfix}.jpg')
    if show:
        plt.show()
    else:
        plt.close()
