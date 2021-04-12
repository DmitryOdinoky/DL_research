

CALL conda.bat activate base


python main.py --learning_rate 1e-4 --batch_size 20 --model ResNet50
python main.py --learning_rate 1e-3 --batch_size 20 --model ResNet50
python main.py --learning_rate 1e-1 --batch_size 20 --model ResNet50

python main.py --learning_rate 1e-4 --batch_size 20 --model ResNet50_drpt
python main.py --learning_rate 1e-3 --batch_size 20 --model ResNet50_drpt
python main.py --learning_rate 1e-1 --batch_size 20 --model ResNet50_drpt

python build_summary.py



