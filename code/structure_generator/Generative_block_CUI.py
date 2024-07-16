#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import EditableMol, BondType
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Descriptors import MolWt
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D, MolDraw2DSVG, MolsToGridImage
from rdkit.Chem.MolStandardize.charge import Uncharger
from itertools import product
from functools import partial
import sys

args = sys.argv


# args[1]: Scaffold SMILES
args.append("CC[N+]1=C(/C=C/C2=C3Oc4ccccc4C=C3CCC2)c2cccc3cccc1c23")
# args[2]: Asterisk indices as strings. Separator is ","
args.append("11,25")
# args[3]: Building block list path
args.append("/home/wusiwei/project/retry/software/data/1.txt")
# args[4]: Input fragment list as strings. Separator is " "
args.append("CCCCCC C")
# args[5]: Priority output fragment SMILES
args.append("")
# args[6]: Output generated molecules SMILES file path
args.append("../../230328_test_generated_mols.csv")


from generator_vae_functions  import  VAE_predict
from util_functions import salt_remover, electron_sanitize, add_asterisk, asterisk_remover, add_scaffold_asterisk, mol_combination, mol_combination_asterisk

# Search similar generated fragments with an input fragment
def next_fragment_selection(Generator):
    fragment_smiles_list = [i for i in Generator.all_generated_fragments]
    fragment_morgan_list = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in fragment_smiles_list]

    fragment_input_smiles = Generator.input_fragment_history[0]
    fragment_input_morgan = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(Generator.start_fragment_smiles), 2, 2048)

    # s = BulkTanimotoSimilarity(fragment_input_morgan, fragment_morgan_list)
    # ss = np.argsort(BulkTanimotoSimilarity(fragment_input_morgan, fragment_morgan_list))
    # sss = np.argsort(BulkTanimotoSimilarity(fragment_input_morgan, fragment_morgan_list))[::-1]

    for i in np.argsort(BulkTanimotoSimilarity(fragment_input_morgan, fragment_morgan_list))[::-1]:
        if fragment_smiles_list[i] not in Generator.input_fragment_history:
            Generator.start_fragment_smiles = fragment_smiles_list[i]
            break
        
    return None

# Estimate Tanimoto similarity from SMILES
def Tanimoto_from_mols(origin_mol, mols_list, return_morgan = False):
    if type(origin_mol) == str:
        origin_mol = Chem.MolFromSmiles(origin_mol)
        mols_list = [Chem.MolFromSmiles(m) for m in mols_list]
        
    origin_morgan = GetMorganFingerprintAsBitVect(origin_mol, 2, 2048)
    mols_morgans = [GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols_list]
    
    # In case of output Morgan fingerprint
    if return_morgan:
        return BulkTanimotoSimilarity(origin_morgan, mols_morgans), origin_morgan, mols_morgans
    
    else:
        return BulkTanimotoSimilarity(origin_morgan, mols_morgans)

class hybrid_generator_multi():
    
    def __init__(self, scaffold_smiles, expand_index, block_list, start_fragment_smiles_list = ["C"]):
        # O=C1NC(=O)c2ccccc21 scaffold
        # 2,7
        # fragment_list
        # CCCCCC C
        
        model_path = "/home/wusiwei/project/retry/software/code/VAE/fragment_VAE.pth"
        label_path = "/home/wusiwei/project/retry/software/code/VAE/fragment_selfies_label.csv"
        
        self.vae = VAE_predict(model_path, label_path)
        self.scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        self.asterisk_scaffold_mol = add_scaffold_asterisk(self.scaffold_mol, expand_index)
        self.scaffold_root = {}
        for atom in self.asterisk_scaffold_mol.GetAtoms():
            if "idx" in atom.GetPropsAsDict():  # GetPropsAsDict获取原子所有属性，idx获取原子索引
                self.scaffold_root[np.int(atom.GetProp("idx"))] = atom.GetIdx() #.GetProp("idx")获取原子索引
        
        self.block_list = []
        for smiles in block_list:
            self.block_list += add_asterisk(smiles)
        
        self.all_generated_mols = {}
        
        self.all_generated_fragments = {}
        
        self.generative_fragments = {}
        self.generative_fragments_asterisk = {}
        
        self.input_fragment_history = [start_fragment_smiles_list]
        
        self.start_fragment_morgan_list = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in self.input_fragment_history[-1]]
        
        
    def mol_with_atom_index(self,mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        return mol

    def one_epoch_generate(self, generate_num = 1000, vae_generate_num = 1000, select_fragment_num = 10000, block_priority = None, asterisk_sel = False):
        
        self.fragment_combination = {}
        
        self.generative_fragments = {}
        
        input_fragment_smiles_list = self.input_fragment_history[-1]

        # Expansion each fragment 
        for fragment_position, input_fragment_smiles in enumerate(self.input_fragment_history[-1]):

            if input_fragment_smiles in self.generative_fragments:
                continue

            else:
                self.fragment_generator(input_fragment_smiles)

                # Attach asterisk to generated fragments
                f_asterisk_list = []
                for s in self.generative_fragments[input_fragment_smiles]:
                    f_asterisk_list += list(set(add_asterisk(s)))          # s一次取一个smiles，f_asterisk_list一次增加多个（在不同的位置上加*号）

                self.generative_fragments_asterisk[input_fragment_smiles] = list(set(f_asterisk_list))

        # Sort by fragment weight
        wt_rank_dict = {}
        for add_idx, input_smiles in enumerate(self.input_fragment_history[-1]):
            
            ml = np.array([Chem.MolFromSmiles(s) for s in self.generative_fragments_asterisk[input_smiles]])
            wt_list = [MolWt(m) for m in ml]  #计算分子量
            wt_list = np.array(wt_list) - MolWt(Chem.MolFromSmiles(input_smiles)) 
            
            sort_array =  np.argsort(np.absolute(wt_list))  #生成分子与输入碎片的分子量差值
            ml = ml[sort_array]
            
            
            # For priority fragment 检查是否含有定制结构
            block_including = []
            if block_priority != None:
                
                for m in ml:
                    block_including.append(len(m.GetSubstructMatch(block_priority)) != 0) #返回GetSubstructMatch子结构匹配的原子索引
                
                wt_rank_dict[self.scaffold_root[add_idx]] = np.concatenate([sort_array[np.array(block_including) ==True], sort_array[np.array(block_including) == False]])
                
            else:
                self.wt_rank_dict[self.scaffold_root[add_idx]] = sort_array

        add_matrix = np.array([list(i) for i in product(list(range(select_fragment_num)), repeat = len(self.scaffold_root))])
        
        generated_mol_list = []
        for i in list(range(np.sum(add_matrix[:-1, :]))):
            print(add_matrix[:-1, :])
            print(np.sum(add_matrix, 1))
            # Combine fragments starting from the smallest combination of fragments 
            for combination_indices in add_matrix[np.sum(add_matrix, 1) == i]:
                add_fragment_list = []
                for root_idx in self.scaffold_root:
                    
                    # Select fragments
                    asterisk_fragment_list = self.generative_fragments_asterisk[self.input_fragment_history[-1][root_idx]]
                    add_fragment_list.append(asterisk_fragment_list[wt_rank_dict[self.scaffold_root[root_idx]][combination_indices[root_idx]]])
                    
                # Fragment addition to the scaffold
                generated_mol_list.append(mol_combination_asterisk(self.asterisk_scaffold_mol, add_fragment_list, self.scaffold_root))
                generated_mol_list = list(set(generated_mol_list))
                    
                # When the specified number of mooecules are generated, exit there.
                if len(generated_mol_list) == generate_num:
                    break
                    
            if len(generated_mol_list) == generate_num:
                break
        
        # Save generated molecules
        for smiles in generated_mol_list:
            self.all_generated_mols[smiles] = None

        return None
        
    # Fragment generation function
    def fragment_generator(self, input_smiles):

        # If start fragment in None, return methyl group and building blocks (not use VAE)
        if input_smiles == None:
            generated_fragment_list = ["*C"] + self.block_list
            self.generative_fragments[input_smiles] = list(set(generated_fragment_list)) 

            return None

        else:                                                            #生成block分子
            # Genrated fragments-store list
            generated_fragment_list = [input_smiles]
            input_fragment_asterisk_list = add_asterisk(input_smiles)

            # Attach a building block to the input fragment comprehensively
            for asterisk_fragment, block in product(input_fragment_asterisk_list, self.block_list):
                generated_fragment_list.append(mol_combination(asterisk_fragment, block))

        # Fragment expansion by VAE
        vae_input_smiles = input_smiles

        input_history = []
        for i in list(range(3)):
            
            input_history.append(vae_input_smiles)

            # Fragments generation by VAE
            vae_generated_fragments = self.vae.part_generator(vae_input_smiles, generate_num = 1000, var_weight = 3)

            # Remove duplications
            vae_generated_fragments = list(set(vae_generated_fragments))

            # Sanitize electron charge and radical
            vae_generated_fragments = electron_sanitize(vae_generated_fragments)

            # Save
            generated_fragment_list += vae_generated_fragments
            generated_fragment_list = list(set(generated_fragment_list))

            # Select next input to VAE
            generted_tanimotos = Tanimoto_from_mols(input_smiles, generated_fragment_list)
            sort_indices = np.argsort(generted_tanimotos)[::-1]

            for idx in sort_indices:
                # Input fragments are not used multiple
                if generated_fragment_list[idx] in input_history:
                    continue

                else:
                    vae_input_smiles = generated_fragment_list[idx]
                    break

        # Save fragment
        self.generative_fragments[input_smiles] = list(set(generated_fragment_list))

        return None 

# Main function

# args[1]: Scaffold SMILES "O=C1NC(=O)c2ccccc21"
args[1] = "CCn1/c(=C/C=C2\\CCCC(/C=C/C3=[N+](CC)c4cccc5cccc3c45)=C2Cl)c2cccc3cccc1c32"
# args[2]: Asterisk indices as strings. Separator is "," "2,7"
args[2] = "20,34"
# args[3]: Building block list path
args[3] = "/home/wusiwei/project/retry/software/data/1.txt"
# args[4]: Input fragment list as strings. Separator is " "
args[4] = "CCC CCC"
# args[5]: Priority output fragment SMILES "C1C2CC1C2"
args[5] = ""
# args[6]: Output generated molecules SMILES file path
args[6] = "/home/wusiwei/project/retry/software/code/structure_generator/210910_test_generated_mols.csv"

block_list = np.array(pd.read_table(args[3], header = None, sep = ",")).tolist()[0][:-1]

if __name__ == "__main__":
    block_list = np.array(pd.read_table(args[3], header = None, sep = ",")).tolist()[0][:-1]
    Generator = hybrid_generator_multi(args[1], [int(i) for i in args[2].split(",")], block_list, [s for s in args[4].split(" ")])
    # Generator.mol_with_atom_index(Generator.asterisk_scaffold_mol)
    priority_structure = Chem.MolFromSmiles(args[5])
    Generator.one_epoch_generate(generate_num = 1000, block_priority = priority_structure, select_fragment_num = 100)
    # next_fragment_selection(Generator)
    print('imok')
    # Output generated molecules
    pd.DataFrame([i for i in Generator.all_generated_mols], columns = ["SMILES"]).to_csv(args[6])


