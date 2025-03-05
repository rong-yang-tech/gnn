# -*- coding: utf-8 -*-
"""
Created on Sun May  8 22:13:30 2022

@author: Administrator
"""

from ase.io import iread
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel, AverageKernel
from sklearn.preprocessing import normalize
import numpy as np
from dscribe.descriptors import MBTR
import sys
from rdkit import Chem
#from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
#import numpy as np
from rdkit.Chem import AllChem
from PIL import Image  # For adjusting DPI

def draw_heat_map(mol1, mol2, att, draw_name):
    mol = Chem.CombineMols(mol1, mol2)
    mol.RemoveAllConformers()
    AllChem.Compute2DCoords(mol)
    # AllChem.Compute2DCoords(mol2)
    Chem.SanitizeMol(mol)
    #mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    #mol = Chem.AddHs(mol)
    # Assume you have a list of attention weights corresponding to each atom in the molecule
    # Replace this with your actual attention weights
    # num_atoms = mol1.GetNumAtoms()
    attention_weights = att  # Example weights
    
    # Normalize the attention weights to [0, 1]
    min_weight = np.min(attention_weights)
    max_weight = np.max(attention_weights)
    norm_weights = (attention_weights - min_weight) / (max_weight - min_weight)  # Add epsilon to avoid division by zero
    #print(norm_weights)
    # Map normalized weights to colors using a colormap
    # norm_weihgt_1, norm_weihgt_2 = norm_weights[:num_atoms], norm_weights[num_atoms:]
    colormap = plt.cm.Greens  # Choose a colormap
    atom_highlight_colors = {}

    num = 0
    for idx, weight in enumerate(norm_weights):
        #print(mol.GetAtoms()[idx].GetAtomicNum())
        
        if mol.GetAtoms()[idx].GetAtomicNum() != 1:
            label = num
            rgba_color = colormap(weight)
            rgb_color = rgba_color[:3]  # RDKit expects RGB tuples
            atom_highlight_colors[label] = rgb_color
            num += 1
        
    # Assign custom labels to atoms using 'atomNote' or 'atomLabel'
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        weight = attention_weights[idx]
        label = f"{weight:.4f}"
        atom.SetProp('atomNote', label)  # Use 'atomNote' to display the label below the atom symbol
        # atom.SetProp('atomLabel', label)  # Use 'atomLabel' if you prefer to replace the atom symbol
    
    # Prepare the molecule for drawing
    mol = Chem.RemoveHs(mol)
    rdMolDraw2D.PrepareMolForDrawing(mol)
    
    inch_size = 10 # Desired physical size in inches
    dpi = 600     # Desired DPI
    pixel_size = int(inch_size * dpi)  # Calculate pixel dimensions
    # Set up the drawing options
    drawer = rdMolDraw2D.MolDraw2DCairo(1500, 1500)
    options = drawer.drawOptions()
    
    # No need to set options.atomLabels
    options.bondLineWidth = 4
    options.atomLabelFontSize = 80
    
    # Highlight atoms without highlighting bonds
    highlight_atoms = list(atom_highlight_colors.keys())
    #print(highlight_atoms)
    
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_highlight_colors,
        highlightBonds=[],               # Do not highlight any bonds
        highlightBondColors={},          # Empty bond colors
        highlightAtomRadii={idx: 0.5 for idx in highlight_atoms}  # Adjust atom highlight radius as needed
    )
    drawer.FinishDrawing()
    image_data = drawer.GetDrawingText()
    # Save the image to a file
    from io import BytesIO
    image = Image.open(BytesIO(image_data))
    image.save('attention/'+ draw_name[:-4] +'.tif', dpi=(dpi, dpi))
    
def soap_desc(path):
    #path = 'traj_3000_3.xyz'
    xyzdata = iread(path)
    desc = SOAP(species=[1, 6, 7, 8], rcut=6.0, nmax=8, lmax=6, average = 'off', periodic=False, crossover=True, sparse=False)
    soap_desc_ls = np.asarray([desc.create(atoms) for atoms in xyzdata])
    print(soap_desc_ls.shape)
    np.save(path.strip('.xyz')+'_soap_'+str(name)+'.npy', soap_desc_ls)
    return soap_desc_ls
    # similarity_mat = np.zeros((len(soap_desc_ls), len(soap_desc_ls)))
    # re = REMatchKernel(metric='linear', alpha=0.00001, threshold=1)
    # #linear = AverageKernel(metric='rbf', gamma=10)
    
    # print(soap_desc_ls.shape)
    # for i in range(len(soap_desc_ls)):
    #     for j in range(i, len(soap_desc_ls)) :
    #         re_kernel = re.create([soap_desc_ls[i], soap_desc_ls[j]])[0][1]
    #         similarity_mat[i][j] = re_kernel
    #         print('#######%d##########%d#######' %(i, j))
    # np.save(path.strip('.xyz')+'.npy', similarity_mat)
def mbtr_desc(path, name): 
    #path = 'traj_3000_3.xyz'
    xyzdata = iread(path)
    desc = MBTR(
    species=["H", "C", "N", "O"],
    k1={
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 0, "max": 8, "n": 100, "sigma": 0.1},
    },
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
    },
    k3={
        "geometry": {"function": "cosine"},
        "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
    },
    periodic=False,
    normalization="l2_each",
    )
    mbtr_desc_ls = np.asarray([desc.create(atoms) for atoms in xyzdata])
    np.save(path.strip('.xyz')+'_mbtr_'+str(name)+'.npy', mbtr_desc_ls)
    return mbtr_desc_ls

class DualOutput():
    """
    print console
    """
    def __init__(self, filename, mode='w', encoding='utf-8'):
        """
        initial DualOutput object。

        :param filename: print file
        :param mode: 'a'。
        :param encoding: 'utf-8'
        """
        self.console = sys.stdout  # save original output
        self.file = open(filename, mode, encoding=encoding)  # open file

    def write(self, message):
        """
        write to console and file

        :param message: str。
        """
        self.console.write(message)  # console
        self.file.write(message)     # file

    def flush(self):
        """
        all output
        """
        self.console.flush()
        self.file.flush()

    def close(self):
        """
        close
        """
        self.file.close()
        
if __name__ == '__main__':
    path = 'traj_all.xyz'
    name = 3500
    soap_desc(path, name)
    #similarity_by_mbtr(path)
    # a = np.load(path.strip('.xyz')+str(name)+'.npy')
    # identity_mat =np.identity(a.shape[0])
    # a_new = a - identity_mat
    # #print(a_new)
    # m = np.where(a_new > 0.99)
    # print(m)
    # all_m = np.concatenate((m[0], m[1]))
    # print(len(set(all_m)))
        
