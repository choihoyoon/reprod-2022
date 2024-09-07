# REPROD's MLOps First Demo

This repository contains first demo version of REPROD's MLOps.

## Pre-requisite

### Server

- CUDA 11.1 or higier
- cuDNN 8.2.1 or higher

### Library

- Python 3.7 or higher
- PyTorch 1.10.1 or higher (You may visit this official PyTorch website for install: https://pytorch.org/)
- tensorflow-gpu 2.9.1 or higher
- keras 2.9.0 or higher
- transformers 4.20.1 or higher

you may also need other packages like,
- pandas
- numpy
- sklearn
- plotly

## Usage

### Clone
```bash
git clone -b master --single-branch https://github.com/nth221/reprod-2022.git
$ cd reprod-2022/code
```

### Run

Below is the example command line for execute the code. We can select hyperparameters via argument inputs.

```bash
python main.py --project demo --task multi-class --model kogpt2 --prj_pth ../project/demo --data_fn ../data/three_demo.csv --split_type auto --input PAPER_TEXT --target RCMN_CD1 --max_length 64 --n_epochs 3 --lr 5e-5 --batch_size 8
```

Below is the information about how to use command line for execute the code.

```bash
$ python main.py --help
usage: main.py [-h] --project PROJECT --task TASK --model MODEL --prj_pth
               PRJ_PTH --data_fn DATA_FN --split_type SPLIT_TYPE
               [--train TRAIN] [--test TEST] --input INPUT --target TARGET
               [--max_length MAX_LENGTH] [--n_epochs N_EPOCHS] [--lr LR]
               [--batch_size BATCH_SIZE]        
```

### Output

The running logs are print out as standard output.
Other artifacts to model learning are save in project folder.

```bash
$ cd project
$ cd {PROJECT_PATH}
$ ls
```
