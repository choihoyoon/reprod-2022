import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import json
from utils import create_directory
from load_data import load_data
from build_model import build_model
from start_train import StratTrain
from strat_optimize import StratOptimize
from write_code import write_code
from write_card import write_card

def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--project', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--model', required=True)

    p.add_argument('--prj_pth', required=True)
    p.add_argument('--data_fn', required=True)

    p.add_argument('--split_type', required=True)
    p.add_argument('--train', type=float, default=.8)
    p.add_argument('--test', type=float, default=.2)
    p.add_argument('--input', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--max_length', type=int, default=32)

    p.add_argument('--auto', type=int, default=0)
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--lr', type=float)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=8)

    config = p.parse_args()

    return config


def create_config_file(config, project_path):
    args_dict = vars(config)
    args_json = json.dumps(args_dict, ensure_ascii=False, indent=4)
    args_file = open(project_path + '/config.json', 'w')
    print(args_json, file=args_file)
    args_file.close()


if __name__ == '__main__':
    config = define_argparser()
    for_card = {}

    project_path = config.prj_pth
    create_directory(project_path)

    data = load_data(config, for_card, project_path)

    if(config.auto == 1):
        StratOptimize(config, project_path, data)

    model = build_model(config, data)
    StratTrain(config, for_card, project_path, data, model)

    create_config_file(config, project_path)
    write_code(config, project_path)
    write_card(config, for_card, project_path)
