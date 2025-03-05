# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:16:52 2024

@author: PC
"""
import re
import glob
import os
import pandas as pd
import shutil
# Define the path to the directory containing the .out files
def delete_folder_recursive(folder_path):
    """DEL"""
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    elif os.path.exists(folder_path):
        os.remove(folder_path)

# PROJECT
directory_path = 'output_psi4/'
 
# RENAME
# RENAME .txt TO .md
def rename_files(directory, old_ext, new_ext):
    for filename in os.listdir(directory):
        if filename.endswith(old_ext):
            old_name = os.path.join(directory, filename)
            new_name = os.path.join(directory, filename.replace(old_ext, new_ext))
            os.rename(old_name, new_name)
 
# .txt to_.md
rename_files(directory_path , '.inp.dat', '.out')        

delete_folder_recursive('initial_data')
os.mkdir('initial_data')
os.mkdir('initial_data/xyz_file')
# Use glob to find all .out files in the specified directory
out_files = glob.glob(os.path.join(directory_path, '*.out'))
file_names = [os.path.basename(file_path) for file_path in out_files]
if 'nohup.out' in file_names:
    file_names.remove('nohup.out')
# Print or process the list of .out files

# Define the path to the input and output files


# Read the content of the input file


# Initialize variables to store fragments

result_dic = {}

for file_name in file_names:
    # print(file_name)
    input_file_path = directory_path+file_name
    output_file_path = ('initial_data/xyz_file/'+file_name).replace('.out', '.xyz')
    dimer_section = []
    inside_dimer = False
    electro_value_kj_mol, exchange_value_kj_mol, induction_value_kj_mol, \
        dispersion_value_kj_mol, total_value_kj_mol = 0,0,0,0,0
    with open(input_file_path, 'r') as file:
        file_content = file.readlines()
    #print(file_content[-1])
    if file_content[-1] == '*** Psi4 exiting successfully. Buy a developer a beer!\n':
    # Parse each line to find the molecule dimer section
        try:
            for line in file_content:
                #print(line)
                if 'molecule dimer {' in line:
                    inside_dimer = True
                    continue  # Skip the line with the opening brace
                if inside_dimer and '}' in line:
                    break  # Stop reading once we reach the closing brace
                if inside_dimer:
                    dimer_section.append(line.strip())
            for  line in file_content:       
                ELECTRO = re.search( r'Electrostatics sSAPT0 .*', line, re.M|re.I)
                #print(ELECTRO)
                if ELECTRO:
                    electro_values = re.findall('-\d+.\d+',ELECTRO.group())
                    if len(electro_values)==0:
                        electro_values = re.findall('\d+.\d+',ELECTRO.group())
                    electro_value_kj_mol = electro_values[-1]
                
                   # print(electro_value_kj_mol)
                ###### 
                EXCHANGE = re.search( r'Exchange sSAPT0 .*', line, re.M|re.I)
                if EXCHANGE:
                    exchange_values = re.findall('-\d+.\d+',EXCHANGE.group())
                    if len(exchange_values)==0:
                        exchange_values = re.findall('\d+.\d+',EXCHANGE.group())
                    exchange_value_kj_mol = exchange_values[-1]
                
                   # print(exchange_value_kj_mol)
                ######
                INDUCTION = re.search( r'Induction sSAPT0 .*', line, re.M|re.I)
                if INDUCTION:
                    induction_values = re.findall('-\d+.\d+',INDUCTION.group())
                    if len(induction_values)==0:
                        induction_values = re.findall('\d+.\d+',INDUCTION.group())
                    induction_value_kj_mol = induction_values[-1]
                
                  #  print(induction_value_kj_mol)
                ######
                DISPERSION = re.search( r'Dispersion sSAPT0 .*', line, re.M|re.I)
                if DISPERSION:
                    dispersion_values = re.findall('-\d+.\d+',DISPERSION.group())
                    if len(dispersion_values)==0:
                        dispersion_values = re.findall('\d+.\d+',DISPERSION.group())
                    dispersion_value_kj_mol = dispersion_values[-1]
                
                   # print(dispersion_value_kj_mol)
                ######
                TOTAL = re.search( r'Total sSAPT0 .*', line, re.M|re.I)
                if TOTAL:
                    total_values = re.findall('-\d+.\d+',TOTAL.group())
                    if len(total_values)==0:
                        total_values = re.findall('\d+.\d+',TOTAL.group())
                    total_value_kj_mol = total_values[-1]
                
                  #  print(total_value_kj_mol)
            key_name = file_name.replace('.out', '.xyz')
            result_dic[key_name] = [electro_value_kj_mol, exchange_value_kj_mol,
                             induction_value_kj_mol, dispersion_value_kj_mol,
                             total_value_kj_mol]
            # Split the section based on the separator '--'
            fragment_1 = []
            fragment_2 = []
            is_second_fragment = False
            
            for line in dimer_section:
                if '--' in line:
                    is_second_fragment = True
                    continue  # Skip the line with the separator
                if not is_second_fragment:
                    fragment_1.append(line)
                else:
                    fragment_2.append(line)
            
            # Helper function to convert fragment data to XYZ format string
            def fragment_to_xyz(fragment):
                coordinates = []
                for line in fragment:
                    parts = line.split()
                    if len(parts) == 4:
                        atom_type, x, y, z = parts
                        coordinates.append(f"{atom_type} {float(x):.6f} {float(y):.6f} {float(z):.6f}")
                xyz_text = f"{len(coordinates)}\nFragment\n" + "\n".join(coordinates)
                return xyz_text
            
            # Convert both fragments to XYZ format and save to the same file
            with open(output_file_path, 'w') as xyz_file:
                xyz_file.write(fragment_to_xyz(fragment_1) + "\n")  # First fragment
                xyz_file.write(fragment_to_xyz(fragment_2) + "\n")     # Second fragment
                xyz_file.write(fragment_to_xyz(fragment_1+fragment_2) + "\n")  # Whole fragment
        except:
            print('wrong!!!')
    else:
        print('xxxxxxxxxxxxxxxxxxxxxxx')
column_name = ['ELECTRO','EXCHANGE','INDUCTION','DISPERISION','TOTAL']
resluts_dataframe = pd.DataFrame(result_dic)
resluts_dataframe = pd.DataFrame(resluts_dataframe.T)
resluts_dataframe.columns = column_name
resluts_dataframe.index = result_dic.keys()
resluts_dataframe.index.name =  'CONFIGURATIONS'
resluts_dataframe.to_csv('initial_data/results.csv')
