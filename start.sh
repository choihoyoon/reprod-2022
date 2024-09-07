#!/bin/sh
export PATH=/home/reprod/bin/miniconda3/envs/py37/bin:$PATH

cd /home/reprod/reprod-2022/code
python main.py --project demo --task multi-class --model kogpt2 --prj_pth ../project/demo --data_fn ../data/three_demo.csv --split_type auto --input PAPER_TEXT --target RCMN_CD1 --max_length 64 --n_epochs 3 --lr 5e-5 --batch_size 8
