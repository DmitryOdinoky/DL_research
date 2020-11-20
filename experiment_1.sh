#!/bin/bash env base


conda activate base

python main_4_classes.py --learning_rate 1e-3 --batch_size 50 --model ResNet50
python main_4_classes.py --learning_rate 1e-3 --batch_size 150 --model ResNet50
python main_4_classes.py --learning_rate 1e-3 --batch_size 200 --model ResNet50
python main_4_classes.py --learning_rate 1e-2 --batch_size 50 --model ResNet50
python main_4_classes.py --learning_rate 1e-2 --batch_size 150 --model ResNet50
python main_4_classes.py --learning_rate 1e-2 --batch_size 200 --model ResNet50
python build_summary.py



