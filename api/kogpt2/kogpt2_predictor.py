import numpy as np
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def class_predict(config, new_sentence, model):
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2',
                                              bos_token='</s>', eos_token='</s>', pad_token='<pad>')

    bos_token = [tokenizer.bos_token]
    eos_token = [tokenizer.eos_token]
    tokens = bos_token + tokenizer.tokenize(new_sentence) + eos_token
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_id = pad_sequences([input_id], maxlen=config['max_length'], value=tokenizer.pad_token_id, padding='post')[0]
    input_id = np.array([input_id])
    score = model.predict(input_id)[0]

    return score.argmax()