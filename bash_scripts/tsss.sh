#!/bin/bash


conda activate conda_env
python main.py --learning_rate 1e-3 --batch_size 200 --model "ResNet50"
python main.py --learning_rate 1e-3 --batch_size 100 --model "ResNet50"
