#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem

class Onehot_Generator:
    """
    Onehot_Generator 类用于对 SELFIES 表示的化学分子进行 One-Hot 编码。
    """
    def __init__(self, input_list, input_type="SELFIES", symbol_label=None, max_length=0):
        """
        初始化 Onehot_Generator 实例。

        参数:
        - input_list: 输入分子列表（SMILES 或 SELFIES 格式）
        - input_type: 输入格式，"SMILES" 或 "SELFIES"（默认）
        - symbol_label: 符号标签列表（可选，默认为 None）
        - max_length: 最大 SELFIES 长度（可选，默认为 0）
        """
        self.input_list = input_list
        self.input_type = input_type
        self.label = []
        self.supplier = []
        self.max_length = max_length
        self.encoded_data = None
        self.invalid_inputs = []  # 记录无效输入
        self.valid_count = 0  # 记录有效输入数量

        # 将 SMILES 转换为 SELFIES（如果需要）
        if input_type == "SMILES":
            self.selfies_list = [self._smiles_to_selfies(smiles, idx) for idx, smiles in enumerate(input_list)]
        else:
            self.selfies_list = input_list

        # 分割 SELFIES 字符串为符号列表
        for idx, selfies in enumerate(self.selfies_list):
            if selfies is None or selfies == "":
                self.supplier.append([])
                self.invalid_inputs.append((idx, f"SELFIES is None or empty, original input: {self.input_list[idx]}"))
                continue
            try:
                # 使用 selfies.split_selfies 分割 SELFIES 字符串
                splitted_selfies = list(sf.split_selfies(selfies))
                self.supplier.append(splitted_selfies)
                self.valid_count += 1
                
                if symbol_label is None:
                    self.label += list(set(splitted_selfies))
            except Exception as e:
                print(f"SELFIES 分割失败 (索引 {idx}): {selfies}, 错误: {e}")
                self.supplier.append([])
                self.invalid_inputs.append((idx, f"SELFIES split failed: {str(e)}, original input: {self.input_list[idx]}"))
        
        # 生成符号标签
        if symbol_label is None:
            # 自动生成符号集，添加填充符 "[_]" 和未知符号 "[Unknown]"
            self.label = sorted(list(set(self.label)))
            self.label = ["[_]"] + self.label + ["[Unknown]"]
            self.label = np.array(self.label)
        else:
            self.label = np.array(symbol_label)
        
        self.max_length = self.length_search()

        # 打印无效输入统计
        if self.invalid_inputs:
            print(f"发现 {len(self.invalid_inputs)} 个无效输入:")
            for idx, error in self.invalid_inputs[:10]:  # 仅打印前 10 个以避免输出过多
                print(f"  索引 {idx}: {error}")
            if len(self.invalid_inputs) > 10:
                print(f"  ... 共 {len(self.invalid_inputs)} 个无效输入")
        print(f"有效输入数量: {self.valid_count}")

    def length_search(self):
        if self.max_length == 0:
            for selfies in self.supplier:
                if self.max_length < len(selfies):
                    self.max_length = len(selfies)
        
        return self.max_length

    def _smiles_to_selfies(self, smiles, idx):
        """
        将 SMILES 字符串转换为 SELFIES 字符串。

        参数:
        - smiles: 输入的 SMILES 字符串
        - idx: 输入索引，用于调试

        返回:
        - selfies: 转换后的 SELFIES 字符串，或 None（如果转换失败）
        """
        if not isinstance(smiles, str) or not smiles.strip():
            print(f"无效输入 (索引 {idx}): {smiles}")
            return None
        smiles = smiles.strip()
        try:
            # 使用 RDKit 验证 SMILES 有效性
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无效 SMILES (索引 {idx}): {smiles}")
                return None
            selfies_str = sf.encoder(smiles)
            return selfies_str
        except Exception as e:
            print(f"SELFIES 转换失败 (索引 {idx}): {smiles}, 错误: {e}")
            return None

    def encoding(self):
        """
        将 SELFIES 符号转换为对应的数字索引。

        返回:
        - encoded_data: 编码后的数据，形状为 (样本数, 最大长度)
        """
        if self.max_length == 0:
            print("警告: max_length 为 0，无法生成编码数据")
            self.encoded_data = np.zeros((len(self.supplier), 0), dtype=int)  # 设置空数组
            return self.encoded_data

        # 初始化编码数组
        self.encoded_data = np.zeros((len(self.supplier), self.max_length), dtype=int)
        
        for mol_idx, mol_symbols in enumerate(self.supplier):
            if not mol_symbols:  # 跳过无效分子
                continue
            for symbol_idx, symbol in enumerate(mol_symbols):
                if symbol in self.label:
                    self.encoded_data[mol_idx, symbol_idx] = np.where(self.label == symbol)[0][0]
                else:
                    # 处理未知符号，使用 "[Unknown]" 的索引
                    self.encoded_data[mol_idx, symbol_idx] = len(self.label) - 1
            
            # 填充剩余部分，使用 "[_]" 的索引 (0)
            self.encoded_data[mol_idx, len(mol_symbols):] = 0
        
        return self.encoded_data
    
    def onehot(self, mode="standard"):
        """
        生成 One-Hot 编码。

        参数:
        - mode: 编码模式，可选 "CNN_input"、"RNN_input" 或 "standard"（默认）

        返回:
        - onehot_data: One-Hot 编码后的数据
        """
        if self.encoded_data is None:
            self.encoding()
        
        if self.max_length == 0 or self.encoded_data.size == 0:
            print("警告: 没有有效数据生成 One-Hot 编码")
            return np.zeros((len(self.supplier), 0, len(self.label)), dtype=np.float32)
        
        # 使用 NumPy 向量化操作生成 One-Hot 编码
        onehot_base = np.eye(len(self.label), dtype=np.float32)[self.encoded_data]
        
        if mode == "CNN_input":
            # (样本数, 特征维度, 符号长度)
            self.onehot_data = onehot_base.transpose(0, 2, 1)
        elif mode == "RNN_input":
            # (符号长度, 样本数, 特征维度)
            self.onehot_data = onehot_base.transpose(1, 0, 2)
        elif mode == "standard":
            # (样本数, 符号长度, 特征维度)
            self.onehot_data = onehot_base
        else:
            raise ValueError("无效的模式! 请使用 'CNN_input'、'RNN_input' 或 'standard'。")
        
        return self.onehot_data

# 测试部分
if __name__ == "__main__":
    try:
        # 使用相对路径读取 CSV 文件
        X = pd.read_csv("../data/250k_rndm_zinc_drugs_clean_3.csv")
        X_sel = X.iloc[:20000, :]
        # 预过滤无效 SMILES
        print("原始 SMILES 数量:", len(X_sel))
        X_sel = X_sel[X_sel["smiles"].apply(lambda x: isinstance(x, str) and x.strip() and Chem.MolFromSmiles(x) is not None)]
        print("过滤后有效 SMILES 数量:", len(X_sel))
        if len(X_sel) == 0:
            print("错误: 没有有效 SMILES 数据，请检查 CSV 文件内容")
        else:
            generator = Onehot_Generator(X_sel["smiles"], input_type="SMILES")
            onehot_data = generator.onehot(mode="standard")
            print(f"One-Hot 编码形状: {onehot_data.shape}")
    except FileNotFoundError:
        print("找不到数据文件，请检查路径 '250k_rndm_zinc_drugs_clean_3.csv'")
    except KeyError as e:
        print(f"CSV 文件缺少 'smiles' 列: {e}")