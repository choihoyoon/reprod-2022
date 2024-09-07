import numpy as np
from tqdm import tqdm


class kobert_feature():
    def __init__(self, max_seq_len, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def convert_examples_to_features(self, examples, labels):
        input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

        for example, label in tqdm(zip(examples, labels), total=len(examples)):
            input_id = self.tokenizer.encode(example, max_length=self.max_seq_len, pad_to_max_length=True)
            padding_count = input_id.count(self.tokenizer.pad_token_id)
            attention_mask = [1] * (self.max_seq_len - padding_count) + [0] * padding_count
            token_type_id = [0] * self.max_seq_len

            assert len(input_id) == self.max_seq_len, "Error with input length {} vs {}".format(len(input_id),
                                                                                                self.max_seq_len)
            assert len(attention_mask) == self.max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), self.max_seq_len)
            assert len(token_type_id) == self.max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_id), self.max_seq_len)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            data_labels.append(label)

        input_ids = np.array(input_ids, dtype=int)
        attention_masks = np.array(attention_masks, dtype=int)
        token_type_ids = np.array(token_type_ids, dtype=int)

        data_labels = np.asarray(data_labels, dtype=np.int32)

        return (input_ids, attention_masks, token_type_ids), data_labels
