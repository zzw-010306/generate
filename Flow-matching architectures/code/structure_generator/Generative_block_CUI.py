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
from itertools import product, islice
import sys
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import torch

# 导入外部模块
from generator_flow_functions import *
from util_functions import salt_remover, electron_sanitize, add_asterisk, asterisk_remover, add_scaffold_asterisk, mol_combination, mol_combination_asterisk

# 搜索与输入片段相似的生成片段
def next_fragment_selection(Generator):
    """根据 Tanimoto 相似性选择与输入片段最相似的未使用片段"""
    fragment_smiles_list = [i for i in Generator.all_generated_fragments]
    fragment_morgan_list = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in fragment_smiles_list]

    fragment_input_smiles = Generator.input_fragment_history[0]
    fragment_input_morgan = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(Generator.start_fragment_smiles), 2, 2048)

    for i in np.argsort(BulkTanimotoSimilarity(fragment_input_morgan, fragment_morgan_list))[::-1]:
        if fragment_smiles_list[i] not in Generator.input_fragment_history:
            Generator.start_fragment_smiles = fragment_smiles_list[i]
            break
    return None


# 定义混合生成器类
class hybrid_generator_multi:
    def __init__(self, scaffold_smiles, expand_index, block_list, start_fragment_smiles_list=["C"], device='cpu'):
        """初始化混合生成器"""
        self.device = torch.device(device)  # 设置设备
        # 初始化 FlowMatchingPredict 模型
        model_path = "path_to_your_model"
        label_path = "fragment_label.csv"
        self.flow = FlowMatchingPredict(model_path, label_path, device=self.device)  # 传递设备参数
        
        # 处理支架分子和扩展点
        self.scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        self.asterisk_scaffold_mol = add_scaffold_asterisk(self.scaffold_mol, expand_index)
        self.scaffold_root = {}
        for atom in self.asterisk_scaffold_mol.GetAtoms():
            if "idx" in atom.GetPropsAsDict():
                self.scaffold_root[int(atom.GetProp("idx"))] = atom.GetIdx()
        
        # 处理构建块
        self.block_list = []
        for smiles in block_list:
            self.block_list += add_asterisk(smiles)
        
        # 初始化存储结构
        self.all_generated_mols = {}
        self.all_generated_fragments = {}
        self.generative_fragments = {}
        self.generative_fragments_asterisk = {}
        self.input_fragment_history = [start_fragment_smiles_list]
        self.start_fragment_morgan_list = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in self.input_fragment_history[-1]]

    def mol_with_atom_index(self, mol):
        """为分子中的每个原子添加索引"""
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        return mol

    def one_epoch_generate(self, generate_num=1000, select_fragment_num=10000, block_priority=None, asterisk_sel=False):
        """生成一轮分子组合"""
        self.fragment_combination = {}
        self.generative_fragments = {}
        input_fragment_smiles_list = self.input_fragment_history[-1]

        # 扩展每个片段
        for fragment_position, input_fragment_smiles in enumerate(input_fragment_smiles_list):
            if input_fragment_smiles not in self.generative_fragments:
                try:
                    self.fragment_generator(input_fragment_smiles)
                except Exception as e:
                    print(f"Error generating fragments for {input_fragment_smiles}: {e}")
                    continue

                f_asterisk_list = []
                for s in self.generative_fragments[input_fragment_smiles]:
                    f_asterisk_list += list(set(add_asterisk(s)))
                self.generative_fragments_asterisk[input_fragment_smiles] = list(set(f_asterisk_list))

        # 根据分子量排序
        wt_rank_dict = {}
        for add_idx, input_smiles in enumerate(input_fragment_smiles_list):
            ml = np.array([Chem.MolFromSmiles(s) for s in self.generative_fragments_asterisk[input_smiles]])
            wt_list = [MolWt(m) for m in ml]
            wt_list = np.array(wt_list) - MolWt(Chem.MolFromSmiles(input_smiles))
            sort_array = np.argsort(np.absolute(wt_list))
            ml = ml[sort_array]

            if block_priority is not None:
                block_including = [len(m.GetSubstructMatch(block_priority)) != 0 for m in ml]
                wt_rank_dict[self.scaffold_root[add_idx]] = np.concatenate([sort_array[np.array(block_including)], sort_array[~np.array(block_including)]])
            else:
                wt_rank_dict[self.scaffold_root[add_idx]] = sort_array

        # 生成分子组合
        add_matrix = product(range(select_fragment_num), repeat=len(self.scaffold_root))
        generated_mol_list = []
        for combination_indices in islice(add_matrix, generate_num):
            try:
                add_fragment_list = [self.generative_fragments_asterisk[input_fragment_smiles_list[root_idx]][wt_rank_dict[self.scaffold_root[root_idx]][idx]] 
                                     for root_idx, idx in enumerate(combination_indices)]
                generated_mol = mol_combination_asterisk(self.asterisk_scaffold_mol, add_fragment_list, self.scaffold_root)
                generated_mol_list.append(generated_mol)
            except Exception as e:
                print(f"Error combining molecule: {e}")
                continue
        
        # 存储生成的分子
        for smiles in generated_mol_list:
            self.all_generated_mols[smiles] = None
        gc.collect()  # 清理内存
        return None

    def fragment_generator(self, input_smiles):
        """生成片段集合"""
        if input_smiles is None:
            generated_fragment_list = ["*C"] + self.block_list
            self.generative_fragments[input_smiles] = list(set(generated_fragment_list))
            return None
        else:
            generated_fragment_set = set([input_smiles])
            input_fragment_asterisk_list = add_asterisk(input_smiles)
            for asterisk_fragment, block in product(input_fragment_asterisk_list, self.block_list):
                generated_fragment_set.add(mol_combination(asterisk_fragment, block))

            # 分批生成片段以优化显存
            batch_size = 100
            total_num = 1000
            for i in range(0, total_num, batch_size):
                try:
                    diffusion_fragments = self.flow.part_generator(input_smiles, generate_num=batch_size)
                    diffusion_fragments = electron_sanitize(list(set(diffusion_fragments)))
                    generated_fragment_set.update(diffusion_fragments)
                except Exception as e:
                    print(f"Error in fragment generation batch {i}: {e}")
                finally:
                    gc.collect()  # 无论是否出错都清理内存

            self.generative_fragments[input_smiles] = list(generated_fragment_set)
            return None

# 主函数
if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:  # 如果没有命令行参数，使用默认测试参数
        args.append("CC[N+]1=C(/C=C/c2ccc(C3=C4Oc5ccccc5C=C4CCC3)s2)c2cccc3cccc1c23")  # args[1]: Scaffold SMILES
        args.append("15,30")  # args[2]: Asterisk indices
        args.append("fragment_list.txt")  # args[3]: Building block list path
        args.append("CCC")  # args[4]: Input fragment list
        args.append("")  # args[5]: Priority output fragment SMILES
        args.append("Cy7-RD_mols.csv")  # args[6]: Output file path

    block_list = np.array(pd.read_table(args[3], header=None, sep=",")).tolist()[0][:-1]
    Generator = hybrid_generator_multi(
        args[1], 
        [int(i) for i in args[2].split(",")], 
        block_list, 
        [s for s in args[4].split(" ")], 
        device='cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    )
    priority_structure = Chem.MolFromSmiles(args[5]) if args[5] else None
    Generator.one_epoch_generate(generate_num=1000, block_priority=priority_structure, select_fragment_num=100)
    print('imok')
    pd.DataFrame([i for i in Generator.all_generated_mols], columns=["SMILES"]).to_csv(args[6])
