import transformers
import pandas as pd
import numpy as np
import os
import urllib.request
from tqdm import tqdm
import tensorflow as tf
from transformers import AutoTokenizer, TFGPT2Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):
    input_ids, data_labels = [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        bos_token = [tokenizer.bos_token]
        eos_token = [tokenizer.eos_token]
        tokens = bos_token + tokenizer.tokenize(example) + eos_token
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_id = pad_sequences([input_id], maxlen=max_seq_len, value=tokenizer.pad_token_id, padding='post')[0]

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        input_ids.append(input_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    data_labels = np.asarray(data_labels, dtype=np.int32)

    return input_ids, data_labels


class TFGPT2ForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name, activation, num_labels):
        super(TFGPT2ForSequenceClassification, self).__init__()
        self.gpt = TFGPT2Model.from_pretrained(model_name, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(0.19592891961278336)
        self.classifier = tf.keras.layers.Dense(num_labels,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation=activation,
                                                name='classifier')

    def call(self, inputs):
        outputs = self.gpt(input_ids=inputs)
        cls_token = outputs[0][:, -1]
        cls_token = self.dropout(cls_token)
        prediction = self.classifier(cls_token)

        return prediction


def main():
    df = pd.read_csv('../data/original.csv')

    target = df[['RCMN_CD1']]
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=777, shuffle=True,
                                             stratify=target)
    target = train_data[['RCMN_CD1']]
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=777, shuffle=True,
                                              stratify=target)

    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2',
                                              bos_token='</s>', eos_token='</s>', pad_token='<pad>')

    max_seq_len = 8
    train_X, train_y = convert_examples_to_features(train_data['PAPER_TEXT'], train_data['RCMN_CD1'],
                                                    max_seq_len=max_seq_len, tokenizer=tokenizer)
    valid_X, valid_y = convert_examples_to_features(valid_data['PAPER_TEXT'], valid_data['RCMN_CD1'],
                                                    max_seq_len=max_seq_len, tokenizer=tokenizer)
    test_X, test_y = convert_examples_to_features(test_data['PAPER_TEXT'], test_data['RCMN_CD1'],
                                                  max_seq_len=max_seq_len, tokenizer=tokenizer)

    if 'multi-class' == 'binary':
        model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2", activation='sigmoid', num_labels=1)
        loss = tf.keras.losses.BinaryCrossentropy()
    elif 'multi-class' == 'multi-class':
        num_labels = df['RCMN_CD1'].nunique()
        model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2", activation='softmax', num_labels=num_labels)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.6266246859806895e-05)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    history = model.fit(train_X, train_y, epochs=4, batch_size= 8,
                        validation_data=(valid_X, valid_y))

    results = model.evaluate(test_X, test_y)
    print("test loss, test acc: ", results)


if __name__ == '__main__':
    main()