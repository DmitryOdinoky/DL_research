from __future__ import print_function, division
import time

from tools import radam
import tools.tensorboard_utils as tensorboard_utils

import datetime

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils


# import os

# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import random

import mmap

import argparse
import os


from tqdm import tqdm

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchnet as tnt

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.utils
from sklearn import datasets

import warnings

from data_class_old import fsd_dataset

warnings.filterwarnings("ignore")

plt.ion()  



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##launch tensorboard via --> tensorboard --logdir=C:/DL_research/DL_research/seq_default




#%%



parser = argparse.ArgumentParser(description='Audio classification example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=str, default='ResNet50', help="model:model_x")

parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')

parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-sequence_name', default='seq_default', type=str)

parser.add_argument('--batch_size', type=int, default=10,
                    help='input batch size for training')

parser.add_argument('--epochs', type=int, default = 10,
                    help='number of epochs to train')

parser.add_argument('--learning_rate', type=float, default=1e-1,
                    help='learning rate for a single GPU')

parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size for convolution')

parser.add_argument('--padding', type=int, default=2,
                    help='padding')

parser.add_argument('--stride', type=int, default=1,
                    help='stride')

parser.add_argument('--classes_amount', type=int, default = 6,
                    help='wtf')

parser.add_argument('--method', type=str, default='mfcc',
                    help='data imaging method')



args = parser.parse_args()

# Checkpoints will be written in the log directory.
args.checkpoint_format = os.path.join(args.log_dir, 'checkpoint-{epoch}.h5')


summary_writer = tensorboard_utils.CustomSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}'
)


#%%

train_dataset = fsd_dataset(csv_file = f'C:/DL_research/FSD/FSDKaggle2018.meta/train_mini_dataset_{args.classes_amount}_class.csv',
                                path = 'C:/DL_research/FSD/FSDKaggle2018.audio_train/',
#     # mmap_path = 'D:/DL_research/DL_research/memmap_stuff/train/memmaps/',
#     # json_path = 'D:/DL_research/DL_research/memmap_stuff/train/json/',                                
                                
                                train = True)




test_dataset = fsd_dataset(csv_file = f'C:/DL_research/FSD/FSDKaggle2018.meta/test_mini_dataset_{args.classes_amount}_class.csv',
                                path = 'C:/DL_research/FSD/FSDKaggle2018.audio_test/',
                                
#     # mmap_path = 'D:/DL_research/DL_research/memmap_stuff/test/memmaps/',
#     # json_path = 'D:/DL_research/DL_research/memmap_stuff/test/json/',                                           

                                train = False)


#%%

# def list_splitter(list_to_split, ratio):
#     elements = len(list_to_split)
#     middle = int(elements * ratio)
#     return [list_to_split[:middle], list_to_split[middle:]]

# random.seed(1)
# train_samples = train_dataset.samples
# # test_samples = test_dataset.samples


# train_samples, test_samples = list_splitter(train_samples, 0.8)

# # test_dataset


# random.seed(1)
# train_dataset.samples = train_samples
# test_dataset.samples = test_samples

#%%



train_loader = torch.utils.data.DataLoader(train_dataset,
                            shuffle=True,
                            batch_size = args.batch_size)

test_loader = torch.utils.data.DataLoader(test_dataset,
                            shuffle=False,
                            batch_size = args.batch_size)




#%%



# train_dataset = fsd_dataset(
#     csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/10_class_MI_big_train.csv',
#     path='D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_train/',
#     # mmap_path = 'D:/DL_research/DL_research/memmap_stuff/train/memmaps/',
#     # json_path = 'D:/DL_research/DL_research/memmap_stuff/train/json/',              
#     train=True)


# test_dataset = fsd_dataset(
#     csv_file='D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/10_class_MI_big_test.csv',
#     path='D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_test/',
#     # mmap_path = 'D:/DL_research/DL_research/memmap_stuff/test/memmaps/',
#     # json_path = 'D:/DL_research/DL_research/memmap_stuff/test/json/',              
#     train=False)

# train_loader = torch.utils.data.DataLoader(train_dataset,
#                             shuffle=True,
#                             batch_size = args.batch_size)

# test_loader = torch.utils.data.DataLoader(test_dataset,
#                             shuffle=False,
#                             batch_size = args.batch_size)









#%%

#torch.manual_seed(1234)

Model = getattr(__import__(f'models_4_audio.{args.model}', fromlist=['Model']), 'Model')
net = Model(args)
net.to(device)


optimizer = radam.RAdam(net.parameters(), lr=args.learning_rate)





    
    
def write_report(var_dict, string):

    postfix = string
    this_time = str(datetime.datetime.now().time()).replace(':','-').replace('.','-')
    this_date = str(datetime.datetime.now().date())
    todays_date = this_date + '_time_'  + this_time[:-7] + '_' + str(args.model)
    
    attributes = list(var_dict.keys())
    index = var_dict['iterationz']
    
    lst = []
    
    for key in var_dict:
    
        lst.append(var_dict[key])
      
    
    transposed = list(map(list, zip(*lst)))
    
    df = pd.DataFrame(transposed ,index=index, columns=attributes)  
    df = df.drop(columns=['iterationz'],axis=0)
    
    
    path = f'D:/DL_research/DL_research/reports_out/{args.method}/'  
    df.to_excel(path + f'output_ver_{todays_date}_{postfix}_BS_{args.batch_size}_lr_{args.learning_rate}_dstype_{args.classes_amount}.xlsx')
    

def make_conf_matrix():
    fig = plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
    plt.xticks(list(train_dataset.decoded_dict.values()), list(train_dataset.decoded_dict.keys()))
    plt.yticks(list(train_dataset.decoded_dict.values()), list(train_dataset.decoded_dict.keys()))
    plt.xticks(rotation=45)
    for x in range(args.classes_amount):
        for y in range(args.classes_amount):
            plt.annotate(
                str(round(100 * conf_matrix[x,y]/np.sum(conf_matrix[x]), 1)), xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='white'
            )
    plt.xlabel('True')
    plt.ylabel('Predicted')
    
    plt.title(f'model: {args.model}, num. of classes: {args.classes_amount}, num. of epochs: {args.epochs}', loc='center')
    
    return fig    
    

number_of_epochs = args.epochs



var_dict = {'iterationz': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'train_f1_scores': [],
            'test_f1_scores': [],
            'train_tnt_loss': [],
            'test_tnt_loss': [],
            #'train_mAPMeter': [],
            #'test_mAPMeter': []
            
            }

meters = {
            'train_tnt_loss': tnt.meter.AverageValueMeter(),
            'test_tnt_loss': tnt.meter.AverageValueMeter(),
            #'train_mAPMeter': tnt.meter.mAPMeter(),
            #'test_mAPMeter': tnt.meter.mAPMeter(),
            # 'train_confusion': tnt.meter.ConfusionMeter(len(train_dataset.labels)),
            # 'test_confusion': tnt.meter.ConfusionMeter(len(train_dataset.labels))


            }

conf_matrix = np.zeros((args.classes_amount, args.classes_amount))


counter = 0
stage = ''




for epoch in range(number_of_epochs):
    
    for key in meters.keys():
        meters[key].reset()
        
              
    counter += 1
    print(f'Epoch #{counter} started')
    
    
    iter_epoch = 0
    
    for loader in [train_loader, test_loader]:
        
        if loader == train_loader:
            stage = 'train'
        else:
            stage = 'test'
            # skip test if the same as train
            if train_loader.dataset == test_loader.dataset:
                break
            
        var_dict_epoch = {
            
            'iterationz': [] ,
            'tnt_loss_epoch': [] ,
            'loss_epoch': [] ,
            'accuracy_epoch': [] ,
            'f1_epoch': []
            
            }
        

        helper = 0
        

        
        for batch in tqdm(loader):
            
            helper += 1
             
            iter_epoch += 1
            
            # print(helper)
            
            images, labels = batch         
            
            
            images, labels = images.to(device), labels.to(device)

            train_X = images
            train_y = labels
            
            optimizer.zero_grad()
            y_prim = net.forward(train_X)
            
            class_count = y_prim.size(1)          
            tmp = torch.arange(class_count).cuda()          

            y = (train_y.unsqueeze(dim=1) == tmp).float() 

            loss = torch.mean(-1*y * torch.log(y_prim) * loader.dataset.y_weights.cuda())
            # loss = torch.mean(-1*y * torch.log(y_prim))
            
            # meters[f'{stage}_tnt_loss'].add(loss.item())
            _, predict_y = torch.max(y_prim, 1)
            
            correct = 0
            total = 0 
            
            for i in range(len(images)):
                act_label = torch.argmax(y[i]) # act_label = 1 (index)
                pred_label = torch.argmax(y_prim[i]) # pred_label = 1 (index)
                
                # conf_matrix[act_label.data.cpu()[i], pred_label.data.cpu()[idx]] += 1

            
                if(act_label == pred_label):
                    correct += 1
                total += 1
            
            accuracy = correct/total
            

            f1 = sklearn.metrics.f1_score(train_y.data.cpu(), predict_y.data.cpu(), average='macro')
            
        

            var_dict_epoch['loss_epoch'].append(loss.item())
            var_dict_epoch['accuracy_epoch'].append(accuracy)
            var_dict_epoch['f1_epoch'].append(f1.item())
            var_dict_epoch['iterationz'].append(iter_epoch)
            
            y_idx = torch.argmax(y, dim=1).data.cpu().numpy()
            y_prim_idx = torch.argmax(y_prim, dim=1).data.cpu().numpy()
            
            for idx in range(y_idx.shape[0]):
                conf_matrix[y_idx[idx], y_prim_idx[idx]] += 1


            if loader == train_loader:
                loss.backward()
                optimizer.step()
                
                      

        print(f"stage: {stage} epoch: {epoch}, Loss: {np.average(var_dict_epoch['loss_epoch'])}, Accuracy: {np.average(var_dict_epoch['accuracy_epoch'])}")

        if stage == 'train':
   
           var_dict[f'{stage}_loss'].append(np.average(var_dict_epoch['loss_epoch']))
           var_dict[f'{stage}_accuracies'].append(np.average(var_dict_epoch['accuracy_epoch']))
           var_dict[f'{stage}_f1_scores'].append(np.average(var_dict_epoch['f1_epoch']))
           
           var_dict[f'{stage}_tnt_loss'].append(meters[f'{stage}_tnt_loss'].value()[0])
           #meters[f'{stage}_mAPMeter'].add(meters[f'{stage}_mAPMeter'].value())

           summary_writer.add_scalar(
                tag=f'{stage}_acc',
                scalar_value=np.average(var_dict_epoch['accuracy_epoch']), 
                global_step=epoch
    )
           
           summary_writer.add_scalar(
                tag=f'{stage}_loss',
                scalar_value=np.average(var_dict_epoch['loss_epoch']), 
                global_step=epoch
    )
           
           summary_writer.add_scalar(
                tag=f'{stage}_f1',
                scalar_value=np.average(var_dict_epoch['f1_epoch']), 
                global_step=epoch
    )

           
           # summary_writer.add_hparams(
           #     hparam_dict=args.__dict__,
           #     metric_dict={
                   
           #         'train_loss': np.average(var_dict_epoch['loss_epoch']),
           #         'train_acc': np.average(var_dict_epoch['accuracy_epoch']),
           #         'train_f1': np.average(var_dict_epoch['f1_epoch'])                   
                   
           #         },
               
           #     name = args.run_name,
           #     global_step=epoch
            
               
           #     )
           
          ##summary_writer.add
             

                                       
        else:
           

            var_dict[f'{stage}_loss'].append(np.average(var_dict_epoch['loss_epoch']))
            var_dict[f'{stage}_accuracies'].append(np.average(var_dict_epoch['accuracy_epoch']))
            var_dict[f'{stage}_f1_scores'].append(np.average(var_dict_epoch['f1_epoch']))
 
            var_dict[f'{stage}_tnt_loss'].append(meters[f'{stage}_tnt_loss'].value()[0])

            
            summary_writer.add_scalar(
                tag=f'{stage}_acc',
                scalar_value=np.average(var_dict_epoch['accuracy_epoch']), 
                global_step=epoch
            )
               
            summary_writer.add_scalar(
                tag=f'{stage}_loss',
                scalar_value=np.average(var_dict_epoch['loss_epoch']), 
                global_step=epoch
                )
            
            summary_writer.add_scalar(
                tag=f'{stage}_f1',
                scalar_value=np.average(var_dict_epoch['f1_epoch']), 
                global_step=epoch
            )
            
            summary_writer.add_hparams(
                hparam_dict=args.__dict__,
                metric_dict={
                
                f'{stage}_loss': np.average(var_dict_epoch['loss_epoch']),
                f'{stage}_acc': np.average(var_dict_epoch['accuracy_epoch']),
                f'{stage}_f1': np.average(var_dict_epoch['f1_epoch'])
                
                },
                
                name = args.run_name,
                global_step=epoch
                 
                
                )
            
            summary_writer.add_figure(
                tag='conf_matrix',
                figure=make_conf_matrix(),
                global_step=epoch
            )
            

            
        # confusion_matrix = meters[f'{stage}_confusion'].value()
        #confusion_matrix = confusion_meter.value() 
        #write_report(var_dict_epoch, f'per_epoch_{stage}')
            
        
   
    
    var_dict['iterationz'].append(counter)
    summary_writer.flush()
    


  
    
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False)

ax1.set_title('1: Loss.  2: Accuracy. 3: F1. ')

sc1 = ax1.plot(var_dict['iterationz'], var_dict['train_loss'], label='train')
sc2 = ax1.plot(var_dict['iterationz'],  var_dict['test_loss'], label='test')


sc3 = ax2.plot(var_dict['iterationz'],  var_dict['train_accuracies'])
sc4 = ax2.plot(var_dict['iterationz'],  var_dict['test_accuracies'])

sc5 = ax3.plot(var_dict['iterationz'], var_dict['train_f1_scores'])
sc6 = ax3.plot(var_dict['iterationz'], var_dict['test_f1_scores'])


ax1.set_xticks([])
ax2.set_xticks([])

leg = ax1.legend(loc='upper right')


fig = make_conf_matrix() # plot one final, at the end



write_report(var_dict, '_')


summary_writer.close()




#%%