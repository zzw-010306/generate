#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import EditableMol, BondType, ReplaceCore, ReplaceSidechains, rdFMCS
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit.Chem.MolStandardize.charge import Uncharger
from itertools import product
import selfies as sf


# Salt remove function
def salt_remover(smiles):
    split_smiles = smiles.split(".")
    sel_idx = np.argmax([len(string) for string in split_smiles])
    return split_smiles[sel_idx]

# Electric charge and radical removing function
def electron_sanitize(smiles_list):
    # Remove electric charge
    uc = Uncharger()
    neutral_mol_list = [uc.uncharge(Chem.MolFromSmiles(s)) for s in smiles_list]

    # Remove fragments including radical
    non_radical_list = []
    for m in neutral_mol_list:

        if "|" in Chem.MolToCXSmiles(m):
            continue

        else:
            non_radical_list.append(Chem.MolToSmiles(m))
            
    return non_radical_list

# Combination making function
def recursive_product(all_list):
    output_combinations = [[element] for element in all_list[0]]
    for list_idx in list(range(1, len(all_list))):
        mid_combinations = []
        for element_output, element_i_list in product(output_combinations, all_list[list_idx]):
            
            mid_combinations.append(element_output + [element_i_list]) 
            
        output_combinations = mid_combinations
        
    return output_combinations

# Comprehensive attach astesrisk on strings
def add_asterisk(smiles_string, origin_string = None):
    asterisk_ring_list = []
    
    # Put asterisk by shiftig one letter at a time
    for idx in list(range(len(smiles_string) + 1)):
        add_asterisk_string = smiles_string[:idx] + "(*)" + smiles_string[idx:]
        asterisk_ring_list.append(add_asterisk_string)
                    
    # Select valid strings and remove duplications 
    asterisk_ring_list = list(set([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in asterisk_ring_list if Chem.MolFromSmiles(s) != None]))
    
    # Select similar structures including asterisks
    if origin_string != None:
        origin_morgan = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(origin_string), 2, 2048)
        asterisk_ring_morgans = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in asterisk_ring_list]
        
        tanimotos = BulkTanimotoSimilarity(origin_morgan, asterisk_ring_morgans)
        
        asterisk_ring_list = [asterisk_ring_list[i] for i in np.argsort(tanimotos)[::-1][:5]]
        
    return asterisk_ring_list

# Asterisk removing function
def asterisk_remover(mol):
    asterisk_idx_list = []
    root_idx_list = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            asterisk_idx_list.append(atom.GetIdx())
            root_idx_list.append(atom.GetNeighbors()[0].GetIdx())

    edit_mol = EditableMol(mol)
    
    for i in list(range(len(asterisk_idx_list))):
        edit_mol.RemoveBond(asterisk_idx_list[i], root_idx_list[i])

    removed_mol = EditableMol.GetMol(edit_mol)
    Chem.SanitizeMol(removed_mol)
    
    removed_mol = Chem.MolFromSmiles(Chem.MolToSmiles(removed_mol).replace("*", "").replace(".", ""))
        
    return removed_mol

# Attach asterisks on the arbitral indices atoms
def add_scaffold_asterisk(scaffold_mol, arbitrary_indices):
    
    add_num = 0
    
    # Convert to list data
    if isinstance(arbitrary_indices, list) == False: # isinstance判断变量类型 返回布尔值
        arbitrary_indices = [arbitrary_indices]

    # Attach asterisks on Mol object and edtit bond by editable mol object 
    for arbitrary_idx in arbitrary_indices:
        asterisk_atom = Chem.MolFromSmiles("*")
        asterisk_atom.GetAtomWithIdx(0).SetProp("idx", str(add_num)) # GetAtomWithIdx根据索引找出对应原子元素
        add_num += 1
        
        edit_combination = EditableMol(Chem.CombineMols(scaffold_mol, asterisk_atom))
        edit_combination.AddBond(arbitrary_idx, len(scaffold_mol.GetAtoms()), BondType.SINGLE)

        # Outputting generated Mol object including asterisk
        scaffold_mol = EditableMol.GetMol(edit_combination)

        
        
    try:
        Chem.SanitizeMol(scaffold_mol)
        
    except:
        return print("Uncorrect scaffold and index!")
    
    return scaffold_mol

# Combined two molecules based on asterisk locations 
def mol_combination(smiles_A, smiles_B):
    # 验证 SMILES
    mol_A = Chem.MolFromSmiles(smiles_A)
    mol_B = Chem.MolFromSmiles(smiles_B)
    if mol_A is None or mol_B is None:
        print(f"Error: Invalid SMILES - smiles_A: {smiles_A}, smiles_B: {smiles_B}")
        return None

    combination_befor_mols = Chem.CombineMols(mol_A, mol_B)

    # 可编辑分子对象
    edit_combination = EditableMol(combination_befor_mols)

    # 查找星号索引
    asterisk_indices_list = []
    for atom in combination_befor_mols.GetAtoms():
        if atom.GetSymbol() == "*":
            asterisk_indices_list.append(atom.GetIdx())
        if len(asterisk_indices_list) == 2:
            break

    # 检查星号数量
    if len(asterisk_indices_list) < 2:
        print(f"Error: Less than two asterisk atoms found - smiles_A: {smiles_A}, smiles_B: {smiles_B}")
        return None

    # 获取邻居索引
    neighbor_idx_1 = combination_befor_mols.GetAtomWithIdx(asterisk_indices_list[0]).GetNeighbors()[0].GetIdx()
    neighbor_idx_2 = combination_befor_mols.GetAtomWithIdx(asterisk_indices_list[1]).GetNeighbors()[0].GetIdx()

    # 编辑键
    edit_combination.RemoveBond(asterisk_indices_list[0], neighbor_idx_1)
    edit_combination.RemoveBond(asterisk_indices_list[1], neighbor_idx_2)
    edit_combination.AddBond(neighbor_idx_1, neighbor_idx_2, BondType.SINGLE)

    # 输出生成的分子
    generated_mol = edit_combination.GetMol()
    Chem.SanitizeMol(generated_mol)
    generated_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(salt_remover(Chem.MolToSmiles(generated_mol))))

    return generated_smiles

def mol_combination_asterisk(asterisk_mol, fragment_smiles_list, scaffold_asterisk_dict):
    
    for position_idx, fragment_smiles in enumerate(fragment_smiles_list):
        asterisk_mol = Chem.CombineMols(asterisk_mol, Chem.MolFromSmiles(fragment_smiles))
        
    # Note the asterisk indices
    asterisk_indices = [atom.GetIdx() for atom in asterisk_mol.GetAtoms() if atom.GetSymbol() == "*"]

    # Editable mol object for structure generations
    edit_combination = EditableMol(asterisk_mol)

    # Attach the fragments in order
    for position_idx in scaffold_asterisk_dict:
        scaffold_asterisk_idx = scaffold_asterisk_dict[position_idx]
        fragment_asterisk_idx = asterisk_indices[len(scaffold_asterisk_dict) + position_idx]
        neighbor_idx_1 = asterisk_mol.GetAtomWithIdx(scaffold_asterisk_idx).GetNeighbors()[0].GetIdx()
        neighbor_idx_2 = asterisk_mol.GetAtomWithIdx(fragment_asterisk_idx).GetNeighbors()[0].GetIdx()

        # Bond edit
        edit_combination.RemoveBond(scaffold_asterisk_idx, neighbor_idx_1)
        edit_combination.RemoveBond(fragment_asterisk_idx, neighbor_idx_2)
        edit_combination.AddBond(neighbor_idx_1, neighbor_idx_2, BondType.SINGLE)

    # Outputting generated Mol object
    generated_mol = EditableMol.GetMol(edit_combination)
    Chem.SanitizeMol(generated_mol)
    generated_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(salt_remover(Chem.MolToSmiles(generated_mol))))

    return generated_smiles
