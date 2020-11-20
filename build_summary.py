import pandas as pd
import numpy as np
import os

#%%

path = 'D:/DL_research/DL_research/reports_out/mfcc/4_classes/'

files = [file for file in os.listdir(path) if file.endswith('.xlsx')]

var_dict = {

            'iterationz': [],
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

container = []

for file in files:
    
    dFrame = pd.read_excel(path + file, index_col=0, comment='#')
    
    dFrame_row = pd.DataFrame(dFrame.iloc[-1])
    
    container.append(dFrame_row)
    
final = pd.concat(container, axis=1).transpose()
final['ind'] = files
final.set_index("ind", inplace = True)

final.to_excel(path + 'SUMMARY.xlsx')

    


