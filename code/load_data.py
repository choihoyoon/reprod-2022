import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertTokenizer
from utils import create_directory
from kobert.kobert_feature import kobert_feature
from kogpt2.kogpt2_feature import kogpt2_feature


def load_data(config, for_card, project_path):
    data_path = project_path + '/data'
    create_directory(data_path)

    df = pd.read_csv(config.data_fn)
    df.to_csv(data_path + '/original.csv')

    for_card['data_size'] = str(os.path.getsize(config.data_fn) / (1024.0 * 1024.0)) + " MB"
    for_card['row'] = df.shape[0]
    for_card['column'] = df.shape[1]
    for_card['num_labels'] = df[config.target].nunique()

    if config.split_type == 'auto':
        config.test = 0.2

    target = df[[config.target]]
    train_data, test_data = train_test_split(df, test_size=config.test, random_state=777, shuffle=True,
                                                 stratify=target)
    target = train_data[[config.target]]
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=777, shuffle=True,
                                                  stratify=target)

    train_data.to_csv(data_path + '/train_data.csv')
    test_data.to_csv(data_path + '/test_data.csv')
    valid_data.to_csv(data_path + '/valid_data.csv')

    max_seq_len = config.max_length
    if config.model == 'kobert':
        tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
        feature_maker = kobert_feature(max_seq_len=max_seq_len, tokenizer=tokenizer)
    elif config.model == 'kogpt2':
        tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2',
                                                  bos_token='</s>', eos_token='</s>', pad_token='<pad>')
        feature_maker = kogpt2_feature(max_seq_len=max_seq_len, tokenizer=tokenizer)

    train_X, train_y = feature_maker.convert_examples_to_features(train_data[config.input], train_data[config.target])
    valid_X, valid_y = feature_maker.convert_examples_to_features(valid_data[config.input], valid_data[config.target])
    test_X, test_y = feature_maker.convert_examples_to_features(test_data[config.input], test_data[config.target])

    np.save(data_path + '/train_X.npy', train_X)
    np.save(data_path + '/train_y.npy', train_y)
    np.save(data_path + '/valid_X.npy', valid_X)
    np.save(data_path + '/valid_y.npy', valid_y)
    np.save(data_path + '/test_X.npy', test_X)
    np.save(data_path + '/test_y.npy', test_y)

    return {'original': df,
            'train_X': train_X, 'train_y': train_y,
            'valid_X': valid_X, 'valid_y': valid_y,
            'test_X': test_X, 'test_y': test_y}
