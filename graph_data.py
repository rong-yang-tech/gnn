"""Data and graphs."""
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset,SubsetRandomSampler
from torch_geometric.data import Dataset, Data, InMemoryDataset
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import os
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_add_pool,TopKPooling,GCNConv,GraphConv,GINConv
att_dtype = np.float32

PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('/RDKit file path**/RDKit/Data/',
                            'BaseFeatures.fdef')  # The 'RDKit file path**' is the installation path of RDKit.
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    




class Graph():
    def __init__(self, rdkit_mol):
        self.atom_type = ['H', 'C', 'N', 'O', 'F',  'Cl', 'Br', 'I']
        self.hybridization = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        self.bond_type = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        #self.smiles = molecule_smiles
        self.atom_pair = self.get_atom_pair_type()
        #print(self.atom_pair)
        self.mol = rdkit_mol
        if self.mol is None:
            self.mol = None
            return

        # Add hydrogens to molecule
        
       # self.mol = Chem.AddHs(self.mol)
        if self.mol is not None:
            self.smiles_to_graph()
    
    def get_atom_pair_type(self):
        
        from itertools import combinations_with_replacement
        
        #elements = ["C", "H", "N", "O", "F", "Cl", "Br", "I"]
        pair_types = ["-".join(sorted(pair)) for pair in combinations_with_replacement(self.atom_type, 2)]
        #print(len(pair_types))
        # 3. ONE-HOT
        pair_to_index = {pair: i for i, pair in enumerate(pair_types)}
        #num_pairs = len(pair_types)
        ##one_hot_vector = np.zeros(num_pairs)
        #index = pair_to_index.get("-".join(sorted(atom_pair)), None)
        #if index is not None:
            #one_hot_vector[index] = 1
        return list(pair_to_index.keys())
        
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            print(x, type(x))
            raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def donor_acceptor(self, rd_mol):
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)
        feats = factory.GetFeaturesForMol(rd_mol)
        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                for u in feats[i].GetAtomIds():
                    is_donor[u] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                for u in feats[i].GetAtomIds():
                    is_acceptor[u] = 1
        return is_donor, is_acceptor

    def AtomAttributes(self, rd_atom, is_donor, is_acceptor, extra_attributes=[]):

        rd_idx = rd_atom.GetIdx()
        # Inititalize
        attributes = []
        # Add atimic number
        attributes += self.one_of_k_encoding(rd_atom.GetSymbol(), self.atom_type)
        # Add heavy neighbor count
        attributes += self.one_of_k_encoding(len(rd_atom.GetNeighbors()), [0, 1, 2, 3, 4, 5, 6])
        # Add neighbor hydrogen count
        attributes += self.one_of_k_encoding(rd_atom.GetTotalNumHs(includeNeighbors=True), [0, 1, 2, 3, 4])
        # Add hybridization type
        attributes += self.one_of_k_encoding(rd_atom.GetHybridization().__str__(), self.hybridization)
        # Add boolean if chiral
        attributes += self.one_of_k_encoding(int(rd_atom.GetChiralTag()), [0, 1, 2, 3])
        # Add boolean if in ring
        attributes.append(rd_atom.IsInRing())
        # Add boolean if aromatic atom
        attributes.append(rd_atom.GetIsAromatic())
        # Add boolean if donor
        attributes.append(is_donor[rd_idx])
        # Add boolean if acceptor
        attributes.append(is_acceptor[rd_idx])

        attributes += extra_attributes
        return np.array(attributes, dtype=att_dtype)

    def atom_featurizer(self, rd_mol):

        is_donor, is_acceptor = self.donor_acceptor(rd_mol)

        #### add atoms descriptors####
        V = []
        for k, atom in enumerate(rd_mol.GetAtoms()):
            all_atom_attr = self.AtomAttributes(atom, is_donor, is_acceptor)
            V.append(all_atom_attr)
        return np.array(V, dtype=att_dtype)
    
    def bond_features(self, bond):
        """

        - bond: RDKit
    

        - features:
        """
        bt = bond.GetBondType()
        #print(bt)
        is_conjugated = bond.GetIsConjugated()
        is_in_ring = bond.IsInRing()
        bond_dict = {
            Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
            Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
            Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
            Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
        }
        bond_feature = bond_dict.get(bt, [0, 0, 0, 0])
        #print(bond_feature)
        # ADDITIONAL
        features = bond_feature + [
            1 if is_conjugated else 0,
            1 if is_in_ring else 0
        ]
        
        
        
        return features
    
    def bond_featurizer(self, mol):


        edge_attrs = []
        #weights = []
       # bond_mat = molDG.GetMoleculeBoundsMatrix(mol)
        #print(bond_mat)
        for a in mol.GetAtoms():  ##ALL ATOMS
            i = a.GetIdx()  ##INDEX
            # print(atom.GetSymbol())  ##
            for neighbor in a.GetNeighbors():
                m = neighbor.GetIdx()  ##INDEX
                # print(neighbor.GetSymbol())  ##SPECIFIC ATOM
                bond_atoms = self.one_of_k_encoding("-".join(sorted([a.GetSymbol(), 
                                           neighbor.GetSymbol()])), self.atom_pair)
                bond = self.mol.GetBondBetweenAtoms(i,m)##
                bond_mat = molDG.GetMoleculeBoundsMatrix(mol)
                bond_length =  bond_mat[i][m]
                s = self.bond_features(bond)
                # print(s)
                edge_attrs.append(bond_atoms + s + [bond_length])
                
        # for bond in mol.GetBonds():
        #     start = bond.GetBeginAtomIdx()
        #     end = bond.GetEndAtomIdx()
        #     #bond_type = bond.GetBondType()
    

        #     edges.append((start, end))
        #     edges.append((end, start))
    

        #     bond_atoms = self.get_one_hot_encoding([mol.GetAtomWithIdx(start).GetSymbol(), 
        #                                mol.GetAtomWithIdx(end).GetSymbol()]).tolist()
            
        #     bond_length =  bond_mat[start][end]
        #     weights.append(bond_length)
            
        #     bond_feature = self.bond_features(bond)
        #     edge_attrs.append(bond_feature + bond_atoms + [bond_length])
        # torch.Tensor
        #edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        #edge_weight = torch.tensor(weights, dtype=torch.float)
        
        return edge_attr #edge_index, edge_attr, edge_weight

    
    def smiles_to_graph(self):
        """
        Converts smiles to a graph.
        """
        #
        #atom_types= ['H','C','N','O','F','Cl','Br','I']
        atoms = self.mol.GetAtoms()

        atoms_list =[]
        for i in atoms:
            atom = i.GetSymbol()##
            atoms_list.append(atom)#LIST
        #print(atoms_list)
        
        # #
        # node_mat = np.zeros((len(atoms_list), len(self.atom_type)))
        # for i, atom in enumerate(atoms_list):
        #     for j, type in enumerate(self.atom_type):
        #         if atom == type:
        #             node_mat[i, j] = 1
        # print(node_mat.shape)
        # REFERENCE node
        node_mat = self.atom_featurizer(self.mol)
        # print(node_mat)

        edge_mat = self.bond_featurizer(self.mol)

         # BOND FEATURE
        # bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        # get_bond = []
        # #print('XXXXXXX')
        # for a in atoms:
        #     i = a.GetIdx()
        #     # print(atom.GetSymbol())
        #     for neighbor in a.GetNeighbors():
        #         m = neighbor.GetIdx()
        #         # print(neighbor.GetSymbol())
        #         bond = self.mol.GetBondBetweenAtoms(i,m)
        #         s = str(bond.GetBondType())
        #         # print(s)
        #         get_bond.append(s)
        # print(get_bond)
        # edge_mat = np.zeros((len(get_bond), len(bond_types)))
        # edge_mat = np.zeros((len(get_bond), len(bond_types)))
        # for l,type in enumerate(get_bond):
        #     if type == 'SINGLE':
        #         edge_mat[l,0] =1
        #     if type == 'DOUBLE':
        #         edge_mat[l,1] =1
        #     if type == 'TRIPLE':
        #         edge_mat[l,2] =1
        #     if type == 'AROMATIC':
        #         edge_mat[l,3] =1
        # print(edge_mat.shape)
        

        # create adjacency matrix
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        # print(adj_mat.shape)
        self.std_adj_mat = np.copy(adj_mat)

        # Create distance matrix
        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1

        # Get modified adjacency matrix
        adj_mat = adj_mat * (1 / dist_mat)

        # Pad the adjacency matrix
        dim_add = len(atoms_list) - adj_mat.shape[0]
        adj_mat = np.pad(
            adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant"
        )

        # Add an identity matrix to adjacency matrix
        # This will make an atom its own neighbor
        adj_mat = adj_mat + np.eye(len(atoms_list))
        # print(adj_mat)


        # # SPARSE MATRIX
        # DEL adj_mat[~np.eye(adj_mat.shape[0], dtype=bool)].reshape(adj_mat.shape[0], -1)
        np.fill_diagonal(adj_mat, 0)
        edge_index = sp.coo_matrix(adj_mat)

        values = edge_index.data  # weight
        indices = np.vstack((edge_index.row, edge_index.col))  # coo
        edge_index = torch.tensor(indices)  # coo



        i = torch.tensor(indices)  # tensor
        v = torch.tensor(values)  # tensor
        edge_index = torch.sparse_coo_tensor(i, v, edge_index.shape)
        edge_index, edge_weight = edge_index._indices(),edge_index._values()

        #edge_index = edge_index.to_dense()
        #edge_index=edge_index.add(edge_index).to_dense()
        #print(edge_index)
        #edge_no1, edge_mat, edge_no2 = self.molecule_to_edge_index_attr_weight(self.mol)
        #print(edge_attr.shape)
        #print(edge_weight)
        # Save both matrices
        self.node_mat = node_mat
        self.edge_mat = edge_mat
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        # print(self.adj_mat)
        # print(self.node_mat)



