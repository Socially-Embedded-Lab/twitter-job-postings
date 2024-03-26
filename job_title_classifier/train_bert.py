import os
import sys
import seaborn as sns
import torch.optim as optim
import transformers
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.job_title_utils import *
from utils.ner_utils import get_hyperparameters


def train(df_texts, df_labels, model, optimizer, num_labels, callbacks, history, tqdm_bar):
    # Training loop
    model.train()
    total_loss = 0
    total_mrr = 0
    for callback_name in callbacks.keys():
        locals()[f'total_{callback_name}'] = 0
    # empty list to save model predictions
    total_preds = []
    postfix = tqdm_bar.postfix
    postfix[7] = len(df_texts) // JobTitleConfig.BATCH_SIZE
    for batch_idx, (texts, labels) in enumerate(zip(np.array_split(df_texts['job_title'],
                                                           len(df_texts) // JobTitleConfig.BATCH_SIZE),
                                            np.array_split(df_labels,
                                                           len(df_texts) // JobTitleConfig.BATCH_SIZE))):
        postfix[6] = batch_idx
        tqdm_bar.postfix = postfix
        tqdm_bar.update()
        # Add batch to gpu
        input_ids = torch.tensor(pad_sequences(
            [tokenizer.encode(text, add_special_tokens=True) for text in list(texts)],
            maxlen=30,
            dtype="long",
            truncating="post",
            padding="post",
        )).to(device)
        attention_masks = torch.tensor(
            [[1 if token != 0 else 0 for token in input_id] for input_id in input_ids]).to(device)
        labels = labels.clone().to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(
            input_ids.long(),
            attention_mask=attention_masks.long(),
            labels=labels.long(),
        )
        loss, preds = outputs[:2]
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
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # model predictions are stored on GPU. So, push it to CPU
        preds = indices.numpy()
        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / postfix[7]
    # avg_accuracy = total_accuracy / len(data_loader)
    avg_mrr = total_mrr / postfix[7]
    for callback_name in callbacks.keys():
        locals()[f'avg_{callback_name}'] = locals()[f'total_{callback_name}'] / postfix[7]
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    # returns the loss and predictions
    history[f'train_losses'].append(avg_loss)
    # history[f'{history_label}_accuracy'].append(avg_accuracy)
    history[f'train_mrr'].append(avg_mrr)
    for callback_name in callbacks.keys():
        history[f'train_{callback_name}'].append(locals()[f'avg_{callback_name}'])
    return avg_loss, history[f'train_accuracy_micro'][-1], total_preds


# function for evaluating the model
def evaluate(df_texts, df_labels, model, num_labels, callbacks, history, tqdm_bar):
    model.eval()
    total_loss = 0
    total_mrr = 0
    for callback_name in callbacks.keys():
        locals()[f'total_{callback_name}'] = 0
    # empty list to save model predictions
    total_preds = []
    if JobTitleConfig.DEBUG:
        print("\nEvaluating...")
    postfix = tqdm_bar.postfix
    postfix[7] = len(df_texts) // JobTitleConfig.BATCH_SIZE

    for batch_idx, (texts, labels) in enumerate(zip(np.array_split(df_texts['job_title'],
                                                           len(df_texts) // JobTitleConfig.BATCH_SIZE),
                                            np.array_split(df_labels,
                                                           len(df_labels) // JobTitleConfig.BATCH_SIZE))):
        postfix[6] = batch_idx
        tqdm_bar.postfix = postfix
        tqdm_bar.update()
        # Add batch to gpu
        input_ids = torch.tensor(pad_sequences(
            [tokenizer.encode(text, add_special_tokens=True) for text in list(texts)],
            maxlen=30,
            dtype="long",
            truncating="post",
            padding="post",
        )).to(device)
        attention_masks = torch.tensor(
            [[1 if token != 0 else 0 for token in input_id] for input_id in input_ids]).to(device)
        labels = labels.clone().to(device)
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids.long(),
                attention_mask=attention_masks.long(),
                labels=labels.long(),
            )
            loss, preds = outputs[:2]
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
            # model predictions are stored on GPU. So, push it to CPU
            preds = indices.numpy()
            # append the model predictions
            total_preds.append(preds)
        # compute the training loss of the epoch
    avg_loss = total_loss / postfix[7]
    # avg_accuracy = total_accuracy / len(data_loader)
    avg_mrr = total_mrr /postfix[7]
    for callback_name in callbacks.keys():
        locals()[f'avg_{callback_name}'] = locals()[f'total_{callback_name}'] / postfix[7]
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    # returns the loss and predictions
    history[f'train_losses'].append(avg_loss)
    # history[f'{history_label}_accuracy'].append(avg_accuracy)
    history[f'train_mrr'].append(avg_mrr)
    for callback_name in callbacks.keys():
        history[f'train_{callback_name}'].append(locals()[f'avg_{callback_name}'])
    return avg_loss, history[f'valid_accuracy_micro'][-1], total_preds


print(f'This run save to: {JobTitleConfig.folder[3:]}\nTraining using {device}')
print(f'Will use: {torch.cuda.get_device_name(0)} as GPU')
makedirs(f"{JobTitleConfig.folder}")
makedirs(f'{JobTitleConfig.folder}/weights')

df = load_data(merge=JobTitleConfig.merge)

examples = df.groupby(f'{JobTitleConfig.level_prefix}_category').nth(5).reset_index().drop('ISCO', axis=1)
examples[
    [f'{JobTitleConfig.level_prefix}_category', 'job_title', f'{JobTitleConfig.level_prefix}_category_label']].to_csv(
    f'{JobTitleConfig.folder}/labels_map.csv', index=False)

labels_map = list(df[f'{JobTitleConfig.level_prefix}_category'].unique())
num_labels = len(labels_map)
print(f'Number of unique labels is: {num_labels}')
time.sleep(1)
df[f'{JobTitleConfig.level_prefix}_category'] = df[f'{JobTitleConfig.level_prefix}_category'].progress_apply(
    lambda cat: labels_map.index(cat))
train_df, test = train_test_split(df, train_size=JobTitleConfig.TRAIN_SIZE,
                                  stratify=df[f'{JobTitleConfig.level_prefix}_category'])
train_df.to_csv(f'{JobTitleConfig.folder}/train.csv', index=False)
test.to_csv(f'{JobTitleConfig.folder}/test.csv', index=False)

train_labels = get_labels(train_df, f'{JobTitleConfig.level_prefix}_category')
test_labels = get_labels(test, f'{JobTitleConfig.level_prefix}_category')

callbacks = get_callbacks(num_labels)
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
# compute the class weights
class_weights = compute_class_weight(None, classes=np.unique(train_labels), y=train_labels.detach().numpy())
# converting list of class weights to a tensor
weights = torch.tensor(class_weights, dtype=torch.float)
# Build model
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels)
model.num_labels = num_labels
model.resize_token_embeddings(len(tokenizer))
# push the model to GPU
model = model.to(device)
# define the optimizer
optimizer_grouped_parameters = get_hyperparameters(model)
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=JobTitleConfig.LR)
# push to GPU
weights = weights.to(device)
# define the loss function
cross_entropy = nn.CrossEntropyLoss(weight=weights)
# set initial loss to infinite
best_valid_loss = float('inf')
best_valid_acc = 0
# empty lists to store training and validation loss of each epoch
history = defaultdict(list)
# history = init_history()
train_loss = valid_loss = train_accuracy = valid_accuracy = best_epoch = 0
epoch = 1
tqdm_postfix = [epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, best_epoch, 0, 0]
# for each epoch
with tqdm(
        bar_format='Epoch: {postfix[0]}/{total} | Training Loss: {postfix[1]:.3f} | Validation Loss: {postfix[2]:.3f} |'
                   ' Training Accuracy: {postfix[3]:.3f} | Validation Accuracy: {postfix[4]:.3f} |'
                   ' Best Epoch: {postfix[5]} | Current Batch: {postfix[6]}/{postfix[7]} | Elapsed: {elapsed} | {rate_fmt}',
        postfix=tqdm_postfix,
        total=JobTitleConfig.NUM_EPOCHS) as t:
    cl = pd.DataFrame()
    for epoch in range(1, JobTitleConfig.NUM_EPOCHS + 1):

        train_loss, train_accuracy, train_preds = train(*shuffle(train_df, train_labels), model, optimizer, num_labels, callbacks,
                                                        history, t)
        t.postfix = [epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, best_epoch, 0, 0]
        t.update()
        valid_loss, valid_accuracy, test_preds = train(test, test_labels, model, optimizer, num_labels, callbacks,
                                                       history, t)
        t.postfix = [epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, best_epoch, 0, 0]
        t.update()
        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_accuracy
            torch.save(model.state_dict(), f'{JobTitleConfig.folder}/weights/saved_weights_{epoch}.pt')
            best_epoch = epoch
        t.postfix = [epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, best_epoch, 0, 0]
        t.update()
        if JobTitleConfig.EARLY_STOPPING and (epoch - best_epoch) == JobTitleConfig.EARLY_STOPPING_THRESHOLD:
            break

# Classification Report
category_labels = list(examples[f'{JobTitleConfig.level_prefix}_category_label'].unique())
cr = classification_report(test_labels, test_preds, output_dict=True)
cr = pd.DataFrame(cr).T
cr = cr.rename({str(i): label for i, label in enumerate(category_labels)})
cr.to_csv(f'{JobTitleConfig.folder}/classification_report.csv')
# Heat Map
cm = confusion_matrix(test_labels, test_preds)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
confusion_matrix_df = pd.DataFrame(data=cm,
                                   columns=examples[f'{JobTitleConfig.level_prefix}_category_label'].unique(),
                                   index=examples[f'{JobTitleConfig.level_prefix}_category_label'].unique())
plt.rcParams["figure.figsize"] = (20, 20)
sns.heatmap(confusion_matrix_df)
plt.title('prediction Heat Map')
plt.savefig(f'{JobTitleConfig.folder}/heat_map.jpg')
plt.show()
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

history = pd.DataFrame(history)
history.index = history.index + 1
history.to_csv(f'{JobTitleConfig.folder}/history.csv', index=False)

x_lim = (1, min(epoch, JobTitleConfig.NUM_EPOCHS) + 20)

plot_stats(history,
           columns=['train_mrr', 'valid_mrr'],
           title='Training and validation MRR', x_label='Epoch', y_label='MRR(Mean Reciprocal Rank)', x_lim=x_lim,
           save_postfix=f'mrr', best_epoch=best_epoch)
plot_stats(history,
           columns=['train_accuracy_micro', 'valid_accuracy_micro', 'train_accuracy_macro', 'valid_accuracy_macro'],
           title='Training and validation Accuracy', x_label='Epoch', y_label='Accuracy', x_lim=x_lim,
           save_postfix=f'accuracy', best_epoch=best_epoch)
plot_stats(history,
           columns=['valid_recall_at_1_macro', 'valid_recall_at_5_macro', 'valid_recall_at_10_macro'],
           title='Validation Recall macro', x_label='Epoch', y_label='Recall', x_lim=x_lim,
           save_postfix=f'macro_recall', best_epoch=best_epoch)
plot_stats(history,
           columns=['valid_recall_at_1_micro', 'valid_recall_at_5_micro', 'valid_recall_at_10_micro'],
           title='Validation Recall micro', x_label='Epoch', y_label='Recall', x_lim=x_lim,
           save_postfix=f'micro_recall', best_epoch=best_epoch)
plot_stats(history,
           columns=['train_losses', 'valid_losses'],
           title='Training and validation Loss', x_label='Epoch', y_label='Loss', x_lim=x_lim,
           save_postfix=f'loss', best_epoch=best_epoch, v_lim=history.valid_losses.max())
