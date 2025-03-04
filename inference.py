# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:21:20 2024

@author: PC
"""

import os

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import  Subset, random_split
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import pandas as pd
from torch_geometric.loader import DataLoader

import sys

from sklearn.metrics import mean_absolute_error,r2_score, mean_absolute_percentage_error
from mol_to_graph import CustomGraphDataset, delete_folder_recursive
from layer_nn import CCPGraph
#import math 
from sklearn.model_selection import KFold
from utils import DualOutput, draw_heat_map

att_dtype = np.float32

#delete_folder_recursive('prediction')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
dataset_name =  'dis_stretch_acetaf'
dataset = CustomGraphDataset(root = dataset_name, mode = 'prediction')
# 'HUBLAU.RFD.CIF_821.xyz','HEYZIX.RFD.cif_737.xyz',
define_draw_name = ['LAZPIO.RFD.cif_421.xyz','LAZPIO.RFD.cif_511.xyz','ACETAF01.RFD.cif_430.xyz','ACETAF01.RFD.cif_566.xyz']

batch_size = 32
soap_truancate = 102
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

if soap_truancate:
    u_dim = [dataset[0].u_soap[:soap_truancate].shape[0], dataset[0].u_dimer.shape[0]]
else:
    u_dim = [dataset[0].u_soap.shape[0], dataset[0].u_dimer.shape[0]]

model = CCPGraph(u_dim, 176, 147, 202,
                 799, 2934, 2997, 1430,
                 726, 0.11111195036105503,0.37688165133524465,0.37558357835099954)
model.load_state_dict(torch.load(dataset_name+'/best_model_train664.pt'))
#print(model)
model.eval()
model.to(device)
config_names = []
inference_results = []
with torch.no_grad():
    for batch_idx, data_batch in enumerate(loader):
        data = data_batch.to(device)
        print(data.name)
        config_names.extend(data.name)
        outputs, att = model(data)
        #print(data.batch)
        # print(outputs)
        unique_indices = torch.unique(data.batch)
    
        # 存储分割后的子张量
        grouped_tensors = []
        
        for idx in unique_indices:
            mask = data.batch == idx
            sub_tensor = att[mask]
            grouped_tensors.append(sub_tensor)
       # label = data.y.unsqueeze(1)
        #loss = criterion(outputs, label)
        #test_losses.append(loss.item())
        # print(outputs.detach().cpu().numpy().shape)
        inference_results.extend(outputs.detach().cpu().numpy())

        #
        # for draw_name in data.name:
        #     # print(draw_name)
        #     for i in define_draw_name:
        #         # print(i)
        #         if draw_name == i:
        #             # print(draw_name)
        #             draw_num = data.name.index(draw_name)
        #             print(data[draw_num].name)
        #             draw_heat_map(data[draw_num].rdkit_mol[0],
        #                           data[draw_num].rdkit_mol[1],
        #                           grouped_tensors[draw_num].cpu().numpy().reshape(-1),
        #                           draw_name)
        # break

        # test_targets.extend(label.numpy())
        # test_predictions.extend(outputs.numpy())
print(config_names, inference_results)
pred_data = pd.DataFrame(inference_results,columns=['ELECTRO', 'EXCHANGE', 'INDUCTION', 'DISPERSION','TOTAL'])
# pred_data = pd.DataFrame({'CONFIGURATIONS':config_names, 'total_energy':inference_results})
pred_data['CONFIGURATIONS'] = config_names
print(pred_data)
pred_data.to_csv(dataset_name + '/prediction_properties.csv')
original_data = pd.read_csv('initial_data/results.csv')
all_data = pd.merge(pred_data, original_data, on = 'CONFIGURATIONS')
all_data.to_csv(dataset_name + '/all_properties.csv')
print(mean_absolute_error(all_data.TOTAL, all_data.total_energy))
print(r2_score(all_data.TOTAL, all_data.total_energy))