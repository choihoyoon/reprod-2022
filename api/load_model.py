import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import transformers
import tensorflow as tf
import json
from build_model import build_model
from kogpt2.kogpt2_predictor import class_predict

def main():
    path = '../project/kogpt_demo'
    config_path = path + '/config.json'

    with open(config_path, 'r') as file:
        config = json.load(file)

    model = build_model(config)
    model.load_weights(path + '/final_model')

    new_sentence_0 = "대장암과 직장암은 각각 대장과 직장의 점막에서 발생하는 악성 종양을 의미합니다. 대장암은 대장 점막이 있는 대장이나 직장의 어느 곳에서나 발생할 수 있지만, S상 결장과 직장에서 가장 자주 생깁니다."
    new_sentence_1 = "텍스트 분류(text classification)는 문자열로 표현된 데이터를 사전에 정해진 분류나 주제, 레이블 등으로 매핑(mapping)하는 자연어 처리 및 기계 학습 파생 데이터 문제를 가리킨다."

    class_predict(config, new_sentence_0, model)
    class_predict(config, new_sentence_1, model)


if __name__ == '__main__':
    main()