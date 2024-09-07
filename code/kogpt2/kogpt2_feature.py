import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


class kogpt2_feature():
    def __init__(self, max_seq_len, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def convert_examples_to_features(self, examples, labels):
        input_ids, data_labels = [], []

        for example, label in tqdm(zip(examples, labels), total=len(examples)):
            bos_token = [self.tokenizer.bos_token]
            eos_token = [self.tokenizer.eos_token]
            tokens = bos_token + self.tokenizer.tokenize(example) + eos_token
            input_id = self.tokenizer.convert_tokens_to_ids(tokens)
            input_id = pad_sequences([input_id], maxlen=self.max_seq_len,
                                     value=self.tokenizer.pad_token_id, padding='post')[0]

            assert len(input_id) == self.max_seq_len, "Error with input length {} vs {}".format(len(input_id),
                                                                                                self.max_seq_len)
            input_ids.append(input_id)
            data_labels.append(label)

        input_ids = np.array(input_ids, dtype=int)
        data_labels = np.asarray(data_labels, dtype=np.int32)

        return input_ids, data_labels
