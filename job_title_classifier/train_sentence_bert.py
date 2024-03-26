import os
from openpyxl import load_workbook
import numpy as np
import plotly.express as px
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler

from config.config import run_prefix, THIS_RUN
from utils.job_title_utils import *
from utils.ner_utils import get_hyperparameters

embedder = JobTitleConfig().embedder

print(f'This run save to: {JobTitleConfig.folder[3:]}\nTraining using {device}')
makedirs(f"{JobTitleConfig.folder}")

df = load_data(merge=JobTitleConfig.merge)

examples = df.groupby(f'{JobTitleConfig.level_prefix}_category').nth(5).reset_index().drop('ISCO', axis=1)
examples[
    [f'{JobTitleConfig.level_prefix}_category', 'job_title',
     f'{JobTitleConfig.level_prefix}_category_label']].to_csv(f'{JobTitleConfig.folder}/labels_map.csv', index=False)

labels_map = list(df[f'{JobTitleConfig.level_prefix}_category'].unique())
num_labels = len(labels_map)
print(f'Number of unique labels is: {num_labels}')
time.sleep(1)
df[f'{JobTitleConfig.level_prefix}_category'] = \
    df[f'{JobTitleConfig.level_prefix}_category'].apply(lambda cat: labels_map.index(cat))
train_df, test = train_test_split(df, train_size=JobTitleConfig.TRAIN_SIZE,
                                  stratify=df[f'{JobTitleConfig.level_prefix}_category'])
train_df.to_csv(f'{JobTitleConfig.folder}/train.csv', index=False)
test.to_csv(f'{JobTitleConfig.folder}/test.csv', index=False)

train_embeddings = get_embeddings(train_df, 'job_title', embedder)
test_embeddings = get_embeddings(test, 'job_title', embedder)

train_labels = get_labels(train_df, f'{JobTitleConfig.level_prefix}_category')
test_labels = get_labels(test, f'{JobTitleConfig.level_prefix}_category')

callbacks = get_callbacks(num_labels)
# wrap tensors
train_data = TensorDataset(train_embeddings, train_labels)
# compute the class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels.detach().numpy())
# converting list of class weights to a tensor
weights = torch.tensor(class_weights, dtype=torch.float)
class_weights_all = class_weights[test_labels]
# sampler for sampling the data during training
train_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

for lr, dropout_rate, regularization_type in tqdm(
        product(JobTitleConfig.LRS, JobTitleConfig.DRS,
                JobTitleConfig.regularization_types),
        total=len(JobTitleConfig.LRS) * len(JobTitleConfig.DRS) * len(JobTitleConfig.regularization_types)):
    regularization = len(regularization_type) > 0
    if (dropout_rate > 1e-3 or lr < 1e-4) and regularization:
        continue
    sub_dir = f"{JobTitleConfig.folder}/lr_{lr}_dr_{dropout_rate}_batch_size_{JobTitleConfig.BATCH_SIZE}_reg_{regularization_type if regularization else 'no'}"
    makedirs(sub_dir)
    makedirs(f'{sub_dir}/weights')
    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=JobTitleConfig.BATCH_SIZE)
    # wrap tensors
    test_data = TensorDataset(test_embeddings, test_labels)
    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_data)
    # dataLoader for test set
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=JobTitleConfig.BATCH_SIZE)
    # Build model
    model = MulticlassClassification(num_labels, dropout_rate=dropout_rate)
    # push the model to GPU
    model = model.to(device)
    # define the optimizer
    optimizer_grouped_parameters = get_hyperparameters(model, JobTitleConfig.FULL_FINE_TUNING)
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)  # learning rate
    # push to GPU
    weights = weights.to(device)
    # define the loss function
    cross_entropy = nn.CrossEntropyLoss(weight=weights)
    loss_func = get_loss_function(model, regularization, regularization_type, cross_entropy)
    # set initial loss to infinite
    best_valid_loss = float('inf')
    best_valid_acc = 0
    # empty lists to store training and validation loss of each epoch
    history = defaultdict(list)
    # history = init_history()
    callback = get_callbacks(num_labels)

    train_loss = valid_loss = train_accuracy = valid_accuracy = best_epoch = best_epoch_loss = best_epoch_acc = 0
    epoch = 1
    tqdm_postfix = [epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, best_epoch]
    # for each epoch
    with tqdm(
            bar_format='Epoch: {postfix[0]}/{total} | Training Loss: {postfix[1]:.3f} | Validation Loss: {postfix[2]:.3f} |'
                       ' Training Accuracy: {postfix[3]:.3f} | Validation Accuracy: {postfix[4]:.3f} |'
                       ' Best Epoch: {postfix[5]} | Elapsed: {elapsed} | {rate_fmt}',
            postfix=tqdm_postfix,
            total=JobTitleConfig.NUM_EPOCHS) as t:
        for epoch in range(1, JobTitleConfig.NUM_EPOCHS + 1):
            # train model
            train_loss, train_accuracy, train_preds = train(train_dataloader, model, optimizer,
                                                            loss_func, num_labels, callback, history)
            t.postfix = [epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, best_epoch]
            t.update()
            # evaluate model
            valid_loss, valid_accuracy, test_preds = evaluate(test_dataloader, model,
                                                              loss_func, num_labels, callback, history)
            # save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch_loss = epoch
            if valid_accuracy > best_valid_acc:
                best_valid_acc = valid_accuracy
                best_epoch_acc = epoch
            if best_epoch_loss == best_epoch_acc == epoch:
                best_epoch = epoch
            torch.save(model.state_dict(), f'{sub_dir}/weights/saved_weights_{epoch}.pt')
            t.postfix = [epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, best_epoch]
            t.update()
            if JobTitleConfig.EARLY_STOPPING and (epoch - best_epoch) == JobTitleConfig.EARLY_STOPPING_THRESHOLD:
                break

    # Classification Report
    category_labels = list(examples[f'{JobTitleConfig.level_prefix}_category_label'].unique())
    cr = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
    cr = pd.DataFrame(cr).T
    cr = cr.rename({str(i): label for i, label in enumerate(category_labels)})
    cr.to_csv(f'{sub_dir}/classification_report.csv')
    # Heat Map
    cm = confusion_matrix(test_labels, test_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    confusion_matrix_df = pd.DataFrame(data=cm,
                                       columns=examples[f'{JobTitleConfig.level_prefix}_category_label'].unique(),
                                       index=examples[f'{JobTitleConfig.level_prefix}_category_label'].unique())
    fig = px.imshow(confusion_matrix_df,
                    color_continuous_scale="Viridis",
                    title='prediction Heat Map'
                    )
    fig.write_html(f'{sub_dir}/heat_map.html')
    history = pd.DataFrame(history)
    history.index = history.index + 1
    history.to_csv(f'{sub_dir}/history.csv', index=False)

    x_lim = (1, min(epoch, JobTitleConfig.NUM_EPOCHS) + 20)

    plot_stats(history,
               columns=['train_mrr', 'valid_mrr'],
               title='Training and validation MRR', x_label='Epoch', y_label='MRR(Mean Reciprocal Rank)', x_lim=x_lim,
               save_postfix=f'mrr', best_epoch=best_epoch, save_dir=sub_dir)
    plot_stats(history,
               columns=['train_accuracy_micro', 'valid_accuracy_micro', 'train_accuracy_macro', 'valid_accuracy_macro'],
               title='Training and validation Accuracy', x_label='Epoch', y_label='Accuracy', x_lim=x_lim,
               save_postfix=f'accuracy', best_epoch=best_epoch, save_dir=sub_dir)
    plot_stats(history,
               columns=['valid_recall_at_1_macro', 'valid_recall_at_5_macro'],  # , 'valid_recall_at_10_macro'
               title='Validation Recall macro', x_label='Epoch', y_label='Recall', x_lim=x_lim,
               save_postfix=f'macro_recall', best_epoch=best_epoch, save_dir=sub_dir)
    plot_stats(history,
               columns=['valid_recall_at_1_micro', 'valid_recall_at_5_micro'],  # , 'valid_recall_at_10_micro'
               title='Validation Recall micro', x_label='Epoch', y_label='Recall', x_lim=x_lim,
               save_postfix=f'micro_recall', best_epoch=best_epoch, save_dir=sub_dir)
    plot_stats(history,
               columns=['train_losses', 'valid_losses'],
               title='Training and validation Loss', x_label='Epoch', y_label='Loss', x_lim=x_lim,
               save_postfix=f'loss', best_epoch=best_epoch, v_lim=history.valid_losses.max(), save_dir=sub_dir)

    best_epoch_row = history.iloc[best_epoch]
    stats = np.array([best_epoch_row.valid_mrr,
                      best_epoch_row.valid_recall_at_1_macro, best_epoch_row.valid_recall_at_1_micro,
                      best_epoch_row.valid_recall_at_5_macro, best_epoch_row.valid_recall_at_5_micro,
                      best_epoch_row.valid_accuracy_macro, best_epoch_row.valid_accuracy_micro])

    workbook_path = f'{run_prefix}job_title_classifier/comparison.xlsx'
    wb = load_workbook(workbook_path)
    page = wb.active
    train_test_split = f'{int(JobTitleConfig.TRAIN_SIZE * 100)}/{int(100 - JobTitleConfig.TRAIN_SIZE * 100)}'
    stats = np.append(stats, stats.mean())
    page.append([THIS_RUN, JobTitleConfig.embedder_name, JobTitleConfig.merge, model.name, num_labels,
                 JobTitleConfig.BATCH_SIZE, lr, dropout_rate, train_test_split, regularization,
                 regularization_type] + list(stats))
    wb.save(filename=workbook_path)
    wb.close()
