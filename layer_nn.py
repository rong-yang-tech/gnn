"""Data and graphs."""
import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from torch_geometric.nn import SAGEConv, global_add_pool,TopKPooling,GCNConv,GraphConv
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_scatter import scatter

from torch_scatter import scatter_add
from torch_geometric.utils import softmax


att_dtype = np.float32

PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('/RDKit file path**/RDKit/Data/',
                            'BaseFeatures.fdef')  # The 'RDKit file path**' is the installation path of RDKit.
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

class Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=43, aggr='sum'):
        super().__init__(aggr=aggr)
        self.aggr = aggr
        self.lin_neg = nn.Linear(in_channels+ edge_dim, out_channels)
        self.lin_root = nn.Linear(in_channels,out_channels)


    def forward(self, x, edge_index, edge_attr):
        #print('x shape:', edge_attr.shape)
        x_adj = torch.cat([x[edge_index[0]], edge_attr], dim=1)
       # print('x shape:',x.shape)
        x_adj = F.tanh(self.lin_neg(x_adj))
        #print('x_adj shape:',x_adj.shape)
        #print('edge shape:',edge_index.shape)
        #x_adj = torch.cat((x_adj, x_adj),0)
        #edge_index_new = torch.cat((edge_index[0],edge_index[1]),0)
        neg_sum = scatter(x_adj, edge_index[0], dim=0, reduce=self.aggr)
        #unique_elements = np.unique(edge_index[1])
        #neg_sum = x_adj.scatter_reduce_(dim=0, index=edge_index, src=x_adj, reduce='sum', include_self=False)
        #print('neg_sum',neg_sum.shape)
        x_out = F.tanh(self.lin_root(x))
        #if x_out.shape[0] != neg_sum.shape[0] :
        #print(x_adj.shape, edge_index.shape, len(unique_elements))
        #print(x_out.shape, neg_sum.shape)
        x_out = x_out + neg_sum
        # x_out = self.bn1(x_out)
        return x_out


"""
############### GlobalAttentaion###############
"""


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        # u=u.view(size,904)
        
        #print('this is u shape {}', {u.shape})



        return out, gate

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'nn={self.nn})')


class CCPGraph(torch.nn.Module):
    
    def __init__(self, u_shape, hidden_dim_1, hidden_dim_2, hidden_dim_3, 
                 hidden_dim_4, hidden_dim_5, hidden_dim_6, hidden_dim_7,
                 hidden_dim_8, dp_rate_1, dp_rate_2, dp_rate_3):
                 
        super().__init__()
        self.u_shape = u_shape
        self.conv1 = Conv(35, hidden_dim_1)
        self.gn1 = GraphNorm(hidden_dim_1)
        self.conv2 =  Conv(hidden_dim_1, hidden_dim_2)
        self.gn2 = GraphNorm(hidden_dim_2)
        self.conv3 =  Conv(hidden_dim_2, hidden_dim_3)
        self.gn3 = GraphNorm(hidden_dim_3)
        # self.conv4 =  Conv(32, 16)
        # self.gn4 = GraphNorm(16)



        # pool
        gate_nn = nn.Sequential(nn.Linear(hidden_dim_3, hidden_dim_4),
                                nn.ReLU(),
                                nn.Linear(hidden_dim_4, hidden_dim_5),
                                nn.ReLU(),
                                nn.Linear(hidden_dim_5, 1))

        self.readout = GlobalAttention(gate_nn)
        self.lin1 = nn.Linear(self.u_shape[0] + self.u_shape[1] + hidden_dim_3, hidden_dim_6)
        # self.lin1 = nn.Linear(hidden_dim_3, hidden_dim_6)


        # self.bn1 = nn.BatchNorm1d(1000)
        self.dp1 = nn.Dropout(p = dp_rate_1)
        self.lin2 = nn.Linear(hidden_dim_6, hidden_dim_7)
        # self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p = dp_rate_2)
        self.lin3 = nn.Linear(hidden_dim_7, hidden_dim_8)
        # self.bn3 = nn.BatchNorm1d(100)
        # self.dp3 = nn.Dropout(p=0.1)
        # self.lin4 = nn.Linear(500, 200)

        # self.bn3 = nn.BatchNorm1d(100)
        self.dp3 = nn.Dropout(p = dp_rate_3)
        self.lin = nn.Linear(hidden_dim_8, 4)
        
        # print('finish')

    def forward(self, data):
        #print(data.x)
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = self.conv2(x, data.edge_index,data.edge_attr)
        x = self.conv3(x, data.edge_index, data.edge_attr)
        # x = self.conv4(x, data.edge_index, data.edge_attr)


        embedding, att = self.readout(x, data.batch)
        size = data.batch[-1].item() + 1
        #print(data.u_soap)
        #u = torch.cat([data.u_soap, data.u_dimer], dim=-1)
        u_soap = data.u_soap.view(size, -1)
        u_dimer = data.u_dimer.view(size, -1)
        u = torch.cat([u_soap[:, :self.u_shape[0]], u_dimer[:, :self.u_shape[1]]], dim=1)
        # u = u_dimer[:, :self.u_shape[1]]

        # SCORE
        # print('xxxxx',att.shape)


        # print('embedding shape',embedding.shape,u,u.shape)
        embedding2 = torch.cat([embedding, u], dim=1)
        #print(embedding.shape, embedding2.shape)

        out = F.relu(self.dp1(self.lin1(embedding2)))
        out = F.relu(self.dp2(self.lin2(out)))
        out = F.relu(self.dp3(self.lin3(out)))

        out_first_4 = self.lin(out)
        
        pred_last = out_first_4.sum(dim=1, keepdim=True)
        out = torch.cat((out_first_4, pred_last), dim=1)
        # print(out)
        # print('finish')
        # return out.view(-1), att
        return out, att






