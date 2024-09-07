from kobert.kobert_model import TFBertForSequenceClassification
from kogpt2.kogpt2_model import TFGPT2ForSequenceClassification
import tensorflow as tf


def build_model(config):
    num_labels = config['num_labels']

    if config['model'] == 'kobert':
        if config['task'] == 'binary':
            model = TFBertForSequenceClassification("klue/bert-base", activation='sigmoid', num_labels=1)
            loss = tf.keras.losses.BinaryCrossentropy()
        elif config['task'] == 'multi-class':
            model = TFBertForSequenceClassification("klue/bert-base", activation='softmax',
                                                                 num_labels=num_labels)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()

    elif config['model'] == 'kogpt2':
        if config['task'] == 'binary':
            model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2", activation='sigmoid',
                                                    num_labels=1, dropout=config['dropout'])
            loss = tf.keras.losses.BinaryCrossentropy()
        elif config['task'] == 'multi-class':
            model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2", activation='softmax',
                                                    num_labels=num_labels, dropout=config['dropout'])
            loss = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
