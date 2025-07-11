#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
import selfies as sf
import sys

# 导入模块
sys.path.append("../flow_matching")
from fragment_flow import FlowMatchingModel
from SELFIES_onehot_generator import Onehot_Generator

# 设备管理函数
def get_device(gpu_index=None):
    if torch.cuda.is_available():
        if gpu_index is not None and gpu_index < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_index}")
        else:
            device = torch.device("cuda")  # 自动选择第一个可用GPU
        print(f"使用设备: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，回退到: CPU")
    return device

class FlowMatchingPredict:
    def __init__(self, model_path, label_path, device='cpu'):
        """初始化 FlowMatchingPredict 类
        Args:
            model_path (str): 预训练模型路径
            label_path (str): SELFIES 符号标签文件路径
            device (str): 计算设备，默认为 'cpu'，可选 'cuda'
        """
        self.device = torch.device(device)
        self.model = FlowMatchingModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.label = list(np.array(pd.read_csv(label_path, index_col=0)["Symbol"]))
        self.max_length = 30
    
    def onehot_encode_from_smiles(self, smiles_list):
        """将 SMILES 列表编码为 one-hot 张量
        Args:
            smiles_list (list): SMILES 字符串列表
        Returns:
            torch.Tensor: one-hot 编码张量
        """
        generator = Onehot_Generator([sf.encoder(smiles) for smiles in smiles_list], 
                                     symbol_label=self.label, max_length=self.max_length)
        return generator.onehot(mode="PyTorch_normal").to(self.device)
    
    def onehot_decode_to_selfies(self, encoded_data):
        """将 one-hot 编码解码为 SELFIES 字符串
        Args:
            encoded_data (torch.Tensor): 形状为 (batch_size, max_length, feature_dim) 的张量
        Returns:
            np.ndarray: SELFIES 字符串数组
        """
        # 如果输入是 logits，先转换为概率分布
        if encoded_data.dim() == 3:  # (batch_size, max_length, feature_dim)
            probs = torch.softmax(encoded_data, dim=-1)
            indices = torch.argmax(probs, dim=-1)  # (batch_size, max_length)
        else:
            indices = encoded_data  # 假设已经是类别索引
        
        restoration_selfies = []
        for i in range(indices.shape[0]):
            symbols = [self.label[idx] for idx in indices[i].tolist()]
            selfies_str = "".join(symbols).replace("[_]", "")
            restoration_selfies.append(selfies_str)
        return np.array(restoration_selfies)
    
    def part_generator(self, input_smiles, generate_num=10, random_state=0):
        """生成新的分子片段
        Args:
            input_smiles (str): 输入的 SMILES 字符串（当前未使用，可作为扩展）
            generate_num (int): 生成的分子数量，默认为 10
            random_state (int): 随机种子，默认为 0
        Returns:
            list: 唯一的有效 SMILES 字符串列表
        """
        # 设置随机种子
        torch.manual_seed(random_state)
        
        # 生成初始样本 x0
        x0 = torch.randn(generate_num, self.max_length, len(self.label), device=self.device)
        
        # 使用 Flow-matching 模型生成样本
        # 假设 forward 方法需要 t0、t1 和 steps 参数
        x1 = self.model(x0, t0=0.0, t1=1.0, steps=10)
        
        # 解码为 SELFIES
        selfies_matrix = self.onehot_decode_to_selfies(x1)
        
        # 转换为 SMILES 并验证
        generated_smiles = []
        for selfies_str in selfies_matrix:
            try:
                smiles = sf.decoder(selfies_str)
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    generated_smiles.append(Chem.MolToSmiles(mol))
            except Exception:
                continue
        
        return list(set(generated_smiles))

# 示例用法
if __name__ == "__main__":
    device = get_device(gpu_index=2)
    predictor = FlowMatchingPredict(
        model_path="../code/result/flow_matching_model_best.pth",
        label_path="fragment_label.csv",
        device=device
    )
    generated = predictor.part_generator(input_smiles="CCO", generate_num=5)
    print("Generated SMILES:", generated)