import sys
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config.config import ConvertConfig
from collections import defaultdict

run_prefix = '' if os.getcwd().endswith('Labour-Market-Statistics') else '../'


def convert_to_iob(input_path, output_file, delimit=',', new_line_sep='\n'):
    prev = None
    prev_tag = None
    count = 0
    multiple = defaultdict(lambda: defaultdict(int))
    with open(input_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimit)
        next(reader, None)
        for i, row in enumerate(reader):
            token, actual_labels = row[-5], row[-4:]
            if token == '':
                continue
            flag = 0
            if i != 0 and row[0] != prev:
                count += 1
                output_file.write(new_line_sep)
                prev_tag = None
            prev = row[0]
            for ind, val in enumerate(actual_labels):
                val = 0 if val == '' else float(val[0])
                if val == 1:
                    flag = 1
                    prefix = 'I' if prev_tag == ConvertConfig.LABELS[ind] and ConvertConfig.iob else 'B'
                    output_file.write(f'{token}\t{prefix}-{ConvertConfig.LABELS[ind]}\n')
                    prev_tag = ConvertConfig.LABELS[ind]
                    if prefix == 'B':
                        multiple[prev][prev_tag] += 1
                    break
            if flag == 0:
                output_file.write(f'{token}\tO\n')
                # prev_tag = None
    return count, sum([dic['LOC'] > 1 for dic in multiple.values()]), sum(
        [dic['JOB_TITLE'] > 1 for dic in multiple.values()])


def split(ratio=0.2):
    df = pd.read_csv(ConvertConfig.job_offer_file)
    ids = list(df['Tweet_ID'].unique())
    train_ids, test_ids = train_test_split(ids, test_size=ratio, random_state=42)
    train = df[df['Tweet_ID'].isin(train_ids)]
    test = df[df['Tweet_ID'].isin(test_ids)]
    print(train[ConvertConfig.LABELS].sum())
    print(test[ConvertConfig.LABELS].sum())
    train.to_csv(ConvertConfig.job_offer_file_train, index=False)
    test.to_csv(ConvertConfig.job_offer_file_test, index=False)


def convert():
    output_train_file = open(ConvertConfig.ner_train_output, "w")
    output_test_file = open(ConvertConfig.ner_test_output, "w")
    split(ratio=0.3)
    train_count, double_loc_count, double_job_count = convert_to_iob(ConvertConfig.job_offer_file_train,
                                                                     output_train_file)
    test_count, test_double_loc_count, test_double_job_count = convert_to_iob(ConvertConfig.job_offer_file_test,
                                                                              output_test_file)
    print(f'Train count = {train_count}, Test count = {test_count}')
    print(f'Train multiple loc count = {double_loc_count}, Test multiple loc count = {test_double_loc_count}')
    print(f'Train multiple job count = {double_job_count}, Test multiple job count = {test_double_job_count}')


if __name__ == "__main__":
    os.chdir('..')
    convert()
