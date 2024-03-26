# Imports
import os
import sys
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from config.config import base_labels

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data.scripts.convert_dataset_to_IOB import convert


# Create class for reading in and separating sentences from their labels
class SentenceGetter(object):
    def __init__(self, data_path, tag2idx):

        """
        Constructs a list of lists for sentences and labels
        from the data_path passed to SentenceGetter.
        We can then access sentences using the .sents
        attribute, and labels using .labels.
        """

        with open(data_path) as f:
            txt = f.read().split("\n\n")

        self.sents_raw = [(sent.split("\n")) for sent in txt]
        self.sents = []
        self.labels = []

        for sent in self.sents_raw:
            tok_lab_pairs = [pair.split() for pair in sent]

            # Handles (very rare) formatting issue causing IndexErrors
            try:
                toks = [pair[0] for pair in tok_lab_pairs]
                labs = [pair[1] for pair in tok_lab_pairs]

                # In the Russian data, a few invalid labels such as '-' were produced
                # by the spaCy preprocessing. Because of that, we generate a mask to
                # check if there are any invalid labels in the sequence, and if there
                # are, we reindex `toks` and `labs` to exclude them.
                mask = [False if l not in tag2idx else True for l in labs]
                if any(mask):
                    toks = list(np.array(toks)[mask])
                    labs = list(np.array(labs)[mask])

            except IndexError:
                continue

            self.sents.append(toks)
            self.labels.append(labs)

        print(f"Constructed SentenceGetter with {len(self.sents)} examples.")


# fmt: on
def get_viz_df(bert_preds_df, spacy_preds_df):
    """
    Joins the prediction dfs and produces the other columns needed for
    comparing/visualizing the models' predictions.
    """

    combined = pd.merge(
        bert_preds_df, spacy_preds_df, left_index=True, right_index=True
    )
    consistency_cols = create_pred_consistency_columns(combined)
    combined = pd.concat([combined, consistency_cols], axis=1)

    for tag in ["per", "loc", "org", "misc"]:
        pred_cols = [f"b_pred_{tag}", f"s_pred_{tag}"]
        combined[f"pred_sum_{tag}"] = combined[pred_cols].sum(axis=1)

    return combined


def create_pred_consistency_columns(combined_df):
    """
    Create columns w/ the model name if one model predicts an
    entity and the other doesn't; otherwise a blank string.
    These columns are used for highlighting the predictions via CSS.
    Condition 1: spaCy predicts no, BERT predicts yes
    Condition 2: spaCy predicts yes, BERT predicts no
    Condition 3: Both models agree
    """

    consistency_cols = []

    for tag in ["per", "loc", "org", "misc"]:
        cond1 = (combined_df[f"s_pred_{tag}"] == 0) & (combined_df[f"b_pred_{tag}"] == 1)
        cond2 = (combined_df[f"s_pred_{tag}"] == 1) & (combined_df[f"b_pred_{tag}"] == 0)
        cond3 = (combined_df[f"b_pred_{tag}"] == 1) & (combined_df[f"s_pred_{tag}"] == 1)

        which_model = np.where(  # noqa
            cond1, "BERT",
            np.where(cond2, "spaCy",
                     np.where(cond3, "", ""))
        )

        consistency_col = pd.Series(which_model, name=f"model_name_{tag}")
        consistency_cols.append(consistency_col)

    return pd.concat(consistency_cols, axis=1)


class BertDataset:
    def __init__(self, sg, tokenizer, max_len, tag2idx):

        """
        Takes care of the tokenization and ID-conversion steps
        for prepping data for BERT.
        Takes a SentenceGetter (sg) initialized on the data you
        want to use as argument.
        """

        pad_tok = tokenizer.vocab["[PAD]"]
        sep_tok = tokenizer.vocab["[SEP]"]
        o_lab = tag2idx["O"]

        # Tokenize the text into subwords in a label-preserving way
        tokenized_texts = [
            tokenize_and_preserve_labels(sent, labs, tokenizer)
            for sent, labs in zip(sg.sents, sg.labels)
        ]

        self.toks = [["[CLS]"] + text[0] for text in tokenized_texts]
        self.labs = [["O"] + text[1] for text in tokenized_texts]

        # Convert tokens to IDs
        self.input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in self.toks],
            maxlen=max_len,
            dtype="long",
            truncating="post",
            padding="post",
        )

        # Convert tags to IDs
        self.tags = pad_sequences(
            [[tag2idx.get(l) for l in lab] for lab in self.labs],
            maxlen=max_len,
            value=tag2idx["O"],
            padding="post",
            dtype="long",
            truncating="post",
        )

        # Swaps out the final token-label pair for ([SEP], O)
        # for any sequences that reach the MAX_LEN
        for voc_ids, tag_ids in zip(self.input_ids, self.tags):
            if voc_ids[-1] == pad_tok:
                continue
            else:
                voc_ids[-1] = sep_tok
                tag_ids[-1] = o_lab

        # Place a mask (zero) over the padding tokens
        self.attn_masks = [[float(i > 0) for i in ii] for ii in self.input_ids]


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def flat_accuracy(valid_tags, pred_tags):
    """
    Define a flat accuracy metric to use while training the model.
    """
    if isinstance(valid_tags[0], list):
        valid_tags = [tag for tags in valid_tags for tag in tags]
        pred_tags = [tag for tags in pred_tags for tag in tags]
    return (np.array(valid_tags) == np.array(pred_tags)).mean()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          save_path=None,
                          cmap=None,
                          normalize=True, show=False):
    if 'O' in target_names:
        target_names = target_names[:-1]
        cm = cm[:-1, :-1]
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(8, 6))
    plt.title(title)

    if target_names is not None:
        if len(target_names) == len(base_labels):
            target_names = [s[2:] for s in target_names]
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def annot_confusion_matrix(valid_tags, pred_tags, epoch, save_path=None, show=False):
    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn `confusion_matrix`.
    """

    if isinstance(valid_tags[0], list):
        valid_tags = [tag for tags in valid_tags for tag in tags]
        pred_tags = [tag for tags in pred_tags for tag in tags]
    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))
    # Calculate the actual confusion matrix
    cm = confusion_matrix(valid_tags, pred_tags, labels=header)
    # header = [label[2:] for label in header]
    plot_confusion_matrix(cm, header, title=f'Confusion matrix - Epoch: {epoch}',
                          save_path=save_path + f'/epoch-{epoch}.jpg', show=show)
    return cm


def process_df():
    convert()


def get_special_tokens(tokenizer, tag2idx):
    pad_tok = tokenizer.vocab["[PAD]"]
    sep_tok = tokenizer.vocab["[SEP]"]
    cls_tok = tokenizer.vocab["[CLS]"]
    o_lab = tag2idx["O"]

    return pad_tok, sep_tok, cls_tok, o_lab


def load_and_prepare_data(config, tokenizer, tag2idx):
    train_data_path = config.train_data_path
    dev_data_path = config.dev_data_path
    max_len = config.MAX_LEN
    batch_size = config.BATCH_SIZE

    getter_train = SentenceGetter(train_data_path, tag2idx)
    getter_dev = SentenceGetter(dev_data_path, tag2idx)
    train = BertDataset(getter_train, tokenizer, max_len, tag2idx)
    dev = BertDataset(getter_dev, tokenizer, max_len, tag2idx)

    # Input IDs (tokens), tags (label IDs), attention masks
    tr_inputs = torch.tensor(train.input_ids)
    val_inputs = torch.tensor(dev.input_ids)
    tr_tags = torch.tensor(train.tags)
    val_tags = torch.tensor(dev.tags)
    tr_masks = torch.tensor(train.attn_masks)
    val_masks = torch.tensor(dev.attn_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=batch_size
    )

    return train_dataloader, valid_dataloader


def get_hyperparameters(model, full_fine_tuning=True):
    if full_fine_tuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters


def print_metrics(metrics):
    for ner_tag in metrics:
        recall = metrics[ner_tag]["TP"] / (metrics[ner_tag]["TP"] + metrics[ner_tag]["FN"])
        precision = metrics[ner_tag]["TP"] / (metrics[ner_tag]["TP"] + metrics[ner_tag]["FP"])
        f1 = 2 * recall * precision / (recall + precision)
        print(f"Label: {ner_tag}, Precision: {precision}, Recall: {recall}, F1: {f1}")


# Calculate metrics
def calculate_metrics(predictions, actual_labels):
    metrics = {}

    LABELS = [
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-JOB_TITLE",
        "I-JOB_TITLE",
        "B-MISC",
        "I-MISC",
        "O"]

    for label in LABELS:
        metrics[label] = {}
        metrics[label]["TP"] = 0.0
        metrics[label]["FP"] = 0.0
        metrics[label]["FN"] = 0.0
        metrics[label]["TN"] = 0.0

    for label, actual_label in zip(predictions, actual_labels):
        if label == actual_label:
            metrics[label]["TP"] += 1
        else:
            metrics[actual_label]["FN"] += 1
            metrics[label]["FP"] += 1
    return metrics


def train_and_save_model(
        model,
        tokenizer,
        optimizer,
        epochs,
        idx2tag,
        tag2idx,
        max_grad_norm,
        device,
        train_dataloader,
        valid_dataloader,
        model_folder,
        figs_folder,
        debug=False,
        show=False
) -> (dict, pd.DataFrame):
    pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, tag2idx)
    history = {
        'Training accuracy': [],
        'Evaluation accuracy': [],
        'Training loss': [],
        'Evaluation loss': [],
    }
    cl = pd.DataFrame()
    epoch = 0
    for _ in trange(epochs, desc="Epoch"):
        epoch += 1

        # Training loop
        model.train()
        tr_loss = 0
        tr_accuracy = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        tr_preds = []
        tr_labels = []

        for _, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Forward pass
            outputs = model(
                b_input_ids.long(),
                attention_mask=b_input_mask.long(),
                labels=b_labels.long(),
            )
            loss, tr_logits = outputs[:2]
            # Backward pass
            loss.backward()
            # Compute train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # Subset out unwanted predictions on CLS/PAD/SEP tokens
            preds_mask = (
                    (b_input_ids != cls_tok)
                    & (b_input_ids != pad_tok)
                    & (b_input_ids != sep_tok)
            )
            tr_logits = tr_logits.detach().cpu().numpy()
            tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            tr_batch_preds = np.argmax(tr_logits[preds_mask.squeeze().cpu()], axis=1)
            tr_batch_labels = tr_label_ids.to("cpu").numpy()
            tr_preds.extend(tr_batch_preds)
            tr_labels.extend(tr_batch_labels)
            # Compute training accuracy
            tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
            tr_accuracy += tmp_tr_accuracy
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )
            # Update parameters
            optimizer.step()

        tr_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        # save accuracy and loss
        history['Training loss'].append(tr_loss)
        history['Training accuracy'].append(tr_accuracy)
        # Validation loop
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels.long(),
                )
                tmp_eval_loss, logits = outputs[:2]

            # Subset out unwanted predictions on CLS/PAD/SEP tokens
            preds_mask = (
                    (b_input_ids != cls_tok)
                    & (b_input_ids != pad_tok)
                    & (b_input_ids != sep_tok)
            )

            logits = logits.detach().cpu().numpy()
            label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            val_batch_preds = np.argmax(logits[preds_mask.squeeze().cpu()], axis=1)
            val_batch_labels = label_ids.to("cpu").numpy()
            predictions.extend(val_batch_preds)
            true_labels.extend(val_batch_labels)

            tmp_eval_accuracy = flat_accuracy(val_batch_labels, val_batch_preds)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        # Evaluate loss, acc, conf. matrix, and class. report on devset
        pred_tags = [idx2tag[i] for i in predictions]
        valid_tags = [idx2tag[i] for i in true_labels]
        cl_report = classification_report([valid_tags], [pred_tags], output_dict=True)
        conf_mat = annot_confusion_matrix(valid_tags, pred_tags, epoch=epoch,
                                          save_path=f'{figs_folder}', show=show)
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        # save accuracy and loss
        history['Evaluation loss'].append(eval_loss)
        history['Evaluation accuracy'].append(eval_accuracy)
        # Save model and optimizer state_dict following every epoch
        save_path = f"{model_folder}/train_checkpoint_epoch_{epoch}.tar"
        torch.save(
            {
                # "epoch": epoch,
                "model_state_dict": model.state_dict(),
                # "optimizer_state_dict": optimizer.state_dict(),
                # "train_loss": tr_loss,
                # "train_acc": tr_accuracy,
                # "eval_loss": eval_loss,
                # "eval_acc": eval_accuracy,
                "classification_report": cl_report,
                "confusion_matrix": conf_mat,
            },
            save_path,
        )
        # Report metrics
        cl_report = pd.DataFrame(cl_report).T.reset_index()
        if debug:
            print(f"Train loss: {tr_loss}")
            print(f"Train accuracy: {tr_accuracy}")
            print(f"Validation loss: {eval_loss}")
            print(f"Validation Accuracy: {eval_accuracy}")
            print(f"Classification Report:\n {cl_report}")
            # print(f"Confusion Matrix:\n {conf_mat}")
            print(f"Checkpoint saved to {save_path}.")
        cl_report['epoch'] = epoch
        cl_report = cl_report.drop('support', axis=1)
        cl = cl.append(cl_report)
    return history, cl
