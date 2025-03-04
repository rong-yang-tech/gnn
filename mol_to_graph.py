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
from graph_data import Graph
#from openbabel import pybel
#import openbabel
from ase.io import iread,write
from dscribe.descriptors import SOAP
from rdkit.Chem import rdDetermineBonds
#from rdkit.Chem import rdMolDescriptors
att_dtype = np.float32
#from io import StringIO
#import pybel
#from rdkit import Chem
#import py3Dmol
#from rdkit.Chem.Draw import IPythonConsole
import shutil
#import gc
#from xyz2mol import xyz2mol, read_xyz_file
import time
import multiprocessing
from sklearn.preprocessing import normalize
from utils import DualOutput
#IPythonConsole.ipython_3d = True
#from xyz2mol import xyz2mol

PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('/RDKit file path**/RDKit/Data/',
                            'BaseFeatures.fdef')  # The 'RDKit file path**' is the installation path of RDKit.
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    
def delete_folder_recursive(folder_path):
    """删除指定路径的文件夹及其下所有文件和子文件夹"""
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    elif os.path.exists(folder_path):
        os.remove(folder_path)
        
from utils import DualOutput        
# ob_log_handler = openbabel.OBMessageHandler()
# ob_log_handler.SetOutputLevel(0)  # 设置为 0 以忽略所有消息
# openbabel.obErrorLog.SetOutputLevel(0)

# 4. 定义获取独热编码的函数

        
class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, mode = 'train', soap_truancate=None,  transform=None, pre_transform=None, pre_filter=None):
        self.soap_truancate = soap_truancate
        self.mode = mode
        super().__init__(root, transform, pre_transform , pre_filter)
        self.root = root
        
        
        #print(self.processed_paths[0])
        self.load(self.processed_paths[0])
        #self.atom_pair = self.get_atom_pair_type()
    
        
    @property
    def raw_file_names(self):
        # 返回原始文件的文件名列表，如果没有原始文件则返回空列表
        return []
    @property
    def processed_file_names(self):
        # 返回预处理文件的文件名列表，如果没有预处理文件则返回空列表
        return ['data.pt']
    
    # @property
    # def soap_truancate(self):
    #     # 返回预处理文件的文件名列表，如果没有预处理文件则返回空列表
    #     return self.soap_truancate

    def download(self):
        # 如果需要，可以在此处下载原始数据
        pass
    
    def get_atom_pair_type(self):
        
        from itertools import combinations_with_replacement
        
        #elements = ["C", "H", "N", "O", "F", "Cl", "Br", "I"]
        atom_type = ['H', 'C', 'N', 'O', 'F',  'Cl', 'Br', 'I']
        pair_types = ["-".join(sorted(pair)) for pair in combinations_with_replacement(atom_type, 2)]
        #print(len(pair_types))
        # 3. 创建独热编码映射表
        pair_to_index = {pair: i for i, pair in enumerate(pair_types)}
        #num_pairs = len(pair_types)
        ##one_hot_vector = np.zeros(num_pairs)
        #index = pair_to_index.get("-".join(sorted(atom_pair)), None)
        #if index is not None:
            #one_hot_vector[index] = 1
        #self.atom_pair = list(pair_to_index.keys())
        return list(pair_to_index.keys())
    
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            print(x, type(x))
            raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))
    
    def calc_soap(self, ase_mol):
        desc = SOAP(species=[1, 6, 7, 8, 9, 17, 35, 53], r_cut=6.0, n_max=6, l_max=4, 
                    average = 'inner', periodic=False,  sparse=False)
        soap_desc = desc.create(ase_mol)
        
        return soap_desc
    
    def cal_distance(self, mol1, mol2):
    
        def species_bond(len1, len2):
            r = 0
            c = 0
            numr = 0
            numc = 0
            for i in dis_new:
                if len1 < abs(i) <= len2:
                    if i < 0:
                        r += i
                        numr += 1
    
                    # for i in dis_new:
                    #     if abs(i) <= 3.5:
                    else:
                        c += i
                        numc += 1
            #
            cbond = c
            rbond = r
            # r 相同键replusion，c不同键connect
            num_bc = numc
            num_br = numr
    
            return rbond, cbond, num_bc, num_br
    
        atoms1 = mol1.get_chemical_symbols()
        atoms2 = mol2.get_chemical_symbols()
        xyzs1 = mol1.get_positions()
        xyzs2 = mol2.get_positions()
        #print(xyzs1)
        #print(xyzs2)
       
        distans = []
        #names_com = []
        atoms_sim = []
    
        for i, atom1 in enumerate(atoms1):
            
            xyz1 = xyzs1[i]
    
            for m, atom2 in enumerate(atoms2):
               
                # print(name1)
                xyz2 = xyzs2[m]
    
                distance = np.linalg.norm(np.array(xyz1) - np.array(xyz2))
                #print(distance)
                distans.append(distance)
                #name_com = name1+name2
                #names_com.append(name_com)
    
                if atom1 == atom2:
                    atom_sim = 0
                else:
                    atom_sim = 1
                atoms_sim.append(atom_sim)
    
    
        zipped = zip(distans, atoms_sim)
        sort_zipped = sorted(zipped, key=lambda x: (x[0]))
        result = zip(*sort_zipped)
        distan_sort, atom_sim_axis = [list(x) for x in result]
        # print(len(distan_sort))
        # print(atom_sim_axis)
    
        # '****统计距离范围键长'
        # #
        dis_new = []
        dis_repul1 = []
        dis_repul2 = []
        dis_repul3 = []
        dis_con1 = []
        dis_con2 = []
        dis_con3 = []
        for i, dis in enumerate(distan_sort):
            num = atom_sim_axis[i]
            if num == 0:
                dis = -dis
                if 1 < abs(dis) < 2:
                    dis_repul1.append(abs(dis))
                if 2 < abs(dis) < 3:
                    dis_repul2.append(abs(dis))
                if 3 < abs(dis) < 4:
                    dis_repul3.append(abs(dis))
            else:
                dis = dis
                if 1 < dis < 2:
                    dis_con1.append(dis)
                if 2 < dis < 3:
                    dis_con2.append(dis)
                if 3 < abs(dis) < 4:
                    dis_con3.append(dis)
    
            dis_new.append(dis)
    
        suppl = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dis_con1.extend(suppl)
        dis_con2.extend(suppl)
        dis_con3.extend(suppl)
        dis_repul1.extend(suppl)
        dis_repul2.extend(suppl)
        dis_repul3.extend(suppl)
    
        rbond0, cbond0, num_bc0, num_br0 = species_bond(1, 2.0)
        rbond1, cbond1,num_bc1,num_br1=species_bond(2.0,2.5)
        rbond2, cbond2, num_bc2, num_br2 = species_bond(2.5, 3)
        rbond3, cbond3, num_bc3, num_br3 = species_bond(3,4)
        numbrc3 = num_bc3 + num_br3
        dis_desc = [num_bc0, num_br0, num_bc1, num_br1, num_bc2, 
                    num_br2 ,numbrc3, dis_repul1[0], dis_repul1[1],
                    dis_repul1[2],dis_repul2[0],dis_repul2[1],dis_repul2[2],
                    dis_repul3[0],dis_repul3[1],dis_repul3[2],dis_repul3[3],
                    dis_repul3[4],dis_repul3[5],dis_repul3[6],dis_repul3[7],
                    dis_repul3[8],dis_repul3[9],dis_repul3[10],dis_repul3[11],
                    dis_con1[0],dis_con1[1],dis_con1[2],dis_con2[0],dis_con2[1],
                    dis_con2[2],dis_con3[0],dis_con3[1],dis_con3[2],dis_con3[3],
                    dis_con3[4],dis_con3[5],dis_con3[6],dis_con3[7],dis_con3[8],
                    dis_con3[9],dis_con3[10],dis_con3[11]]
        
        return dis_desc
    
    def ase_to_rdkit(self, atoms):
        import subprocess, re
        symbols = atoms.get_chemical_symbols()
        if len(symbols) < 100:
            write('temp.xyz', atoms)
            # atoms, charge, xyz_coordinates = read_xyz_file('temp.xyz')
            # rdkit_mol = xyz2mol(atoms, xyz_coordinates, charge=charge)
            byte_str = subprocess.check_output(['obabel', '-ixyz', 'temp.xyz', '-osmi'])#
            decoded_str = byte_str.decode('utf-8')
            pattern_simple = r'^(.*?)\t'
            smiles = re.findall(pattern_simple, decoded_str, re.MULTILINE)[0]
            #mol.addh()
            #smiles = mol.write('smiles').strip()
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            Chem.SanitizeMol(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG()) 
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s', maxIters=1000)
            #mol = Chem.RemoveHs(mol)
            
                
            return mol
        else:
            print('atom number by pass 30')
            return False
    
    
    def calculate_inertia_angle_between_atoms(self, atoms1, atoms2):
        """
        计算两个 ASE Atoms 对象的主惯性轴之间的夹角（度数）。
        
        参数:
        atoms1 -- 第一个 ASE Atoms 对象
        atoms2 -- 第二个 ASE Atoms 对象
        
        返回:
        angles -- 一个列表，包含每对主惯性轴之间的夹角（度数）
        """
        # 定义内部函数来计算主惯性轴
        def get_inertia_tensor(atoms):
            positions = atoms.get_positions()
            masses = atoms.get_masses()
            center_of_mass = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
            positions -= center_of_mass  # 平移到质心坐标系
            
            Ixx = np.sum(masses * (positions[:, 1]**2 + positions[:, 2]**2))
            Iyy = np.sum(masses * (positions[:, 0]**2 + positions[:, 2]**2))
            Izz = np.sum(masses * (positions[:, 0]**2 + positions[:, 1]**2))
            Ixy = -np.sum(masses * positions[:, 0] * positions[:, 1])
            Ixz = -np.sum(masses * positions[:, 0] * positions[:, 2])
            Iyz = -np.sum(masses * positions[:, 1] * positions[:, 2])
            
            inertia_tensor = np.array([[Ixx, Ixy, Ixz],
                                       [Ixy, Iyy, Iyz],
                                       [Ixz, Iyz, Izz]])
            return inertia_tensor
        
        def get_principal_axes(atoms):
            inertia_tensor = get_inertia_tensor(atoms)
            _, eigenvectors = np.linalg.eigh(inertia_tensor)
            return eigenvectors  # 主惯性轴的方向（列向量）
    
        # 获取两个 Atoms 对象的主惯性轴
        axes1 = get_principal_axes(atoms1)
        axes2 = get_principal_axes(atoms2)
        
        # 计算每对对应主惯性轴的夹角
        angles = []
        for i in range(3):
            dot_product = np.dot(axes1[:, i], axes2[:, i])
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 夹角防止超出[-1, 1]范围
            angles.append(np.degrees(angle))  # 转换为角度
        
        return angles
    
    def find_closest_atoms(self, atoms1, atoms2):
        # GET ATOMS
        symbols1 = atoms1.get_chemical_symbols()
        positions1 = atoms1.get_positions()
        
        symbols2 = atoms2.get_chemical_symbols()
        positions2 = atoms2.get_positions()
        
        # MINI
        min_distance = float('inf')
        closest_pair = (None, None)  #  (symbol1, symbol2)
        
        # Atoms
        for i, (symbol1, pos1) in enumerate(zip(symbols1, positions1)):
            for j, (symbol2, pos2) in enumerate(zip(symbols2, positions2)):
                # DIS
                distance = np.linalg.norm(pos1 - pos2)
                
                # <MIN,
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (symbol1, symbol2)

        # 输出最近邻原子对的类型和距离
        #print(f"最近邻原子对: {closest_pair[0]} - {closest_pair[1]}")
        #print(f"最短距离: {min_distance:.4f} Å")
        #print(type(np.array([min_distance])))
        
        return self.one_of_k_encoding("-".join(sorted(closest_pair)), self.get_atom_pair_type()), [min_distance], \
               self.calculate_inertia_angle_between_atoms(atoms1, atoms2)

    
    def process(self):
        # PROCESS，SAVE UNDER self.processed_dir


        # df = pd.read_csv('tesall_data_rdx.csv')
        if self.mode == 'train':
            df = pd.read_csv('initial_data/results.csv').iloc[:, :]
            # y = df["tot"].to_list()
            # x = []
            # edge_feats = []
            # adj_mat = []
            data_list=[]
            out_list = []
            wrong_mol = []
            not_even_mol = []
            for index, row in df.iterrows():
                try:
                    print(index)
                    config = row["CONFIGURATIONS"]
                    xyz_path = 'initial_data/xyz_file/'+config

                    # print(xyz_path)
                    ase_mol = [i for i in iread(xyz_path)]
                    # print(ase_mol[2])
                    pair_type, pair_distance, angle = self.find_closest_atoms(ase_mol[0], ase_mol[1])
                    config_atom_type_distance = self.cal_distance(ase_mol[0], ase_mol[1])

                    u_soap = torch.tensor(self.calc_soap(ase_mol[2]), dtype=torch.float)


                    u_dimer = torch.tensor(pair_type + config_atom_type_distance + angle, dtype=torch.float)
                    
                    
                    # u=F.normalize(u,)
                    
                    y = [row["ELECTRO"], row["EXCHANGE"], row["INDUCTION"],
                         row["DISPERISION"], row["TOTAL"]]
                    
                    y= torch.tensor(y,dtype=torch.float)
                    #print(u)
                    # Create MolGraph object
                    #print('OOOOOOOOOOOOOOK')
                    a1 = self.ase_to_rdkit(ase_mol[0])
                    #print(a1)
                    if a1 :
                        mol1 = Graph(a1)
                    else :
                        
                        out_list.append(config)
                        continue
                    a2 = self.ase_to_rdkit(ase_mol[1])
                    #print(a2)
                    # print(mol1.node_mat)
                    if a2 :
                        mol2 = Graph(a2)
                    else :
                        out_list.append(config)
                        continue
                        
                        
                    #print('OOOOOOOOOOOOOOK2222')
                    
                    node_mat_tot = np.concatenate([mol1.node_mat, mol2.node_mat], axis=0)
                    node_mat_tot = torch.tensor(node_mat_tot,dtype=torch.float)
                    edge_mat_tot = np.concatenate([mol1.edge_mat, mol2.edge_mat], axis=0)
                    edge_mat_tot = torch.tensor(edge_mat_tot,dtype=torch.float)
                  
                    inputs=[mol1.edge_index, mol2.edge_index+mol1.node_mat.shape[0]]
                    edge_index = torch.cat(inputs, dim=1)
                    edge_weight = torch.cat([mol1.edge_weight, mol2.edge_weight], dim=0)
                    
                    data = Data(x=node_mat_tot, edge_attr=edge_mat_tot, 
                                edge_index=edge_index, edge_weight=edge_weight, 
                                y=y, u_soap=u_soap, u_dimer = u_dimer, 
                                name = config, rdkit_mol = [a1, a2])       # 根据需要创建图数据
                  
                    data_list.append(data)
                    print('*'*30)
                   
                except Exception as inner_error:
                    print('problem!!!!, this is %s' %(config))
                    print(inner_error)
                    
                    wrong_mol.append(config)
                    print('*'*30)
            #print(data_list)
            #data,slices = self.collate(data_list)
            print('These are moelcules with atom number by pass 30', out_list)
            print('These are moelcules with errors',wrong_mol)
            print('These are moelcules with no even atoms number',not_even_mol)
            self.save(data_list, self.processed_paths[0])
            
        elif self.mode == 'prediction':
            import glob
            folder_path = 'dis_stretch_xyz_acetaf'
            file_list = glob.glob(os.path.join(folder_path, '*.xyz'))

            data_list=[]
            wrong_mol = []
            for file in file_list:
                try:
                    print(file)
                    config = file[len(folder_path)+1:]
                    print(config)
                    ase_mol = [i for i in iread(file)]
                    print('asemol',ase_mol[0])
                    pair_type, pair_distance, angle = self.find_closest_atoms(ase_mol[0], ase_mol[1])
                    config_atom_type_distance = self.cal_distance(ase_mol[0], ase_mol[1])
                   
                    u_soap = torch.tensor(self.calc_soap(ase_mol[2]), dtype=torch.float)
                   
                    
                    u_dimer = torch.tensor(pair_type + config_atom_type_distance + angle, dtype=torch.float)
                    
                    
                    # u=F.normalize(u,)
                    
                    #print(u)
                    # Create MolGraph object
                    #print('OOOOOOOOOOOOOOK')
                    a1 = self.ase_to_rdkit(ase_mol[0])
                    #print(a1)
                    if a1 :
                        mol1 = Graph(a1)
                    else :
                        
                        out_list.append(config)
                        continue
                    a2 = self.ase_to_rdkit(ase_mol[1])
                    #print(a2)
                    # print(mol1.node_mat)
                    if a2 :
                        mol2 = Graph(a2)
                    else :
                        out_list.append(config)
                        continue
                        
                        
                        
                    #print('OOOOOOOOOOOOOOK2222')
                    node_mat_tot = np.concatenate([mol1.node_mat, mol2.node_mat], axis=0)
                    node_mat_tot = torch.tensor(node_mat_tot,dtype=torch.float)
                    edge_mat_tot = np.concatenate([mol1.edge_mat, mol2.edge_mat], axis=0)
                    edge_mat_tot = torch.tensor(edge_mat_tot,dtype=torch.float)
                  
                    inputs=[mol1.edge_index, mol2.edge_index+mol1.node_mat.shape[0]]
                    edge_index = torch.cat(inputs, dim=1)
                    edge_weight = torch.cat([mol1.edge_weight, mol2.edge_weight], dim=0)
                    
                    data = Data(x=node_mat_tot, edge_attr=edge_mat_tot, 
                                edge_index=edge_index, edge_weight=edge_weight, 
                                u_soap=u_soap, u_dimer = u_dimer, 
                                name = config, rdkit_mol = [a1, a2])       # 根据需要创建图数据
                  
                    data_list.append(data)
                    print('*'*30)
                except Exception as inner_error:
                    print('problem!!!!, this is %s' %(config))
                    print(inner_error)
                    
                    wrong_mol.append(config)
                    print('*'*30)
            self.save(data_list, self.processed_paths[0])
if __name__ == '__main__':
    
    import seaborn as sns
    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    # from sklearn.metrics.pairwise import cosine_similarity
    # from sklearn.preprocessing import MinMaxScaler
    from dscribe.kernels import REMatchKernel
    
    def draw_hot_map(dataset):
        smiles_list = [data.name[:-4] for data in dataset]
        fingerprints = [data.u_soap.numpy()[:] for data in dataset]

        num_mols = len(fingerprints)
        # print(fingerprints[0].shape)
        similarity_matrix = np.zeros((num_mols, num_mols))
        print(similarity_matrix)
        re = REMatchKernel(metric="rbf", alpha=1, threshold=1e-6, gamma=1000)
        #scaler = MinMaxScaler()
        #descriptor_array = scaler.fit_transform(fingerprints)

        # def generalized_tanimoto_similarity(v1, v2):
        #     numerator = np.dot(v1, v2)
        #     denominator = np.sum(v1**2) + np.sum(v2**2) - numerator
        #     if denominator == 0:
        #         return 0
        #     else:
        #         return numerator / denominator
        
        # SIMILARITY MATRIX
        #num_mols = len(descriptor_array)
        # similarity_matrix = np.zeros((num_mols, num_mols))
        
        for i in range(num_mols):
            for j in range(num_mols):
                similarity =  re.create([normalize([fingerprints[i]]), normalize([fingerprints[j]])])
                print(similarity)
                similarity_matrix[i, j] = similarity[0][1]
        
        # matrix  DataFrame
        similarity_df = pd.DataFrame(similarity_matrix, index=smiles_list, columns=smiles_list)
        
        # heatmap
        plt.figure(figsize=(16, 8),dpi=150)
        sns.heatmap(similarity_df,
                    annot=True,
                    fmt=".2f",
                    cmap='Greens',
                    xticklabels=True,
                    yticklabels=True,
                    vmin=0, vmax=1,)
        # plt.title(' Generalized Tanimoto Similarity')
        plt.xlabel('Dimer',fontsize=14)
        plt.ylabel('Dimer',fontsize=14)
        plt.show()
        
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)

        # off_diagonal
        similarity_values_off_diagonal = similarity_matrix[mask]
        
        # off_diagonal
        mean_off_diagonal = np.mean(similarity_values_off_diagonal)
        std_off_diagonal = np.std(similarity_values_off_diagonal)
        print(len(dataset[0].u_soap.numpy()))
        #print(len(fingerprints[0]))
        print(f"mean_off_diagonal：{mean_off_diagonal:.4f}")
        print(f"std_off_diagonal：{std_off_diagonal:.4f}")
        data = dataset[1]
        #print('XXX', Chem.MolToSmiles(data.rdkit_mol[0]))
        print('XXX', data.u_soap.numpy())
        sns.lineplot(x= range(len(data.u_soap.numpy())), y = data.u_soap.numpy())    
    
    dual_output = DualOutput('log_graph.txt')
    sys.stdout = dual_output
    mode_and_path = 'train_acetaf'
    
    # delete_folder_recursive(mode_and_path)
    
    dataset = CustomGraphDataset(root = mode_and_path, mode = 'train')

    sys.stdout = dual_output.console
    dual_output.close()
    draw_hot_map(dataset)
    
