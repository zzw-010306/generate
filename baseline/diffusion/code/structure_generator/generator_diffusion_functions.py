#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
import selfies as sf
import sys
# 导入扩散模型
sys.path.insert(0, "../diffusion")
from fragment_diffusion import DiffusionModel, UNet
from SELFIES_onehot_generator import Onehot_Generator

class Diffusion_predict():
    def __init__(self, model_path, label_path):
        self.model = DiffusionModel(UNet())
        # 从模型参数中获取设备
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.model = DiffusionModel(UNet()).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.label = list(np.array(pd.read_csv(label_path, index_col=0)["Symbol"]))
        self.max_length = 30
        self.input_dim = len(self.label)

    # SMILES转独热编码
    def onehot_encode_from_smiles(self, smiles_list):
        generator = Onehot_Generator([sf.encoder(smiles) for smiles in smiles_list], symbol_label=self.label, max_length=self.max_length)
        return generator.onehot(mode="PyTorch_normal")

    # 独热编码转SELFIES
    def onehot_decode_to_selfies(self, encoded_data):
        restoration_selfies = []
        for i in range(encoded_data.shape[0]):
            indices = torch.argmax(encoded_data[i], dim=-1).cpu().numpy()
            restoration_selfies.append("".join(np.array(self.label)[indices].tolist()).replace("[_]", ""))
        return np.array(restoration_selfies)

    # 使用扩散模型生成分子
    def part_generator(self, input_smiles, generate_num=10, random_state=0):
        np.random.seed(random_state)
        # device = torch.device("cpu")

        # 从纯噪声开始生成样本
        generated_data = self.model.sample(batch_size=generate_num, max_length=self.max_length, feature_dim=self.input_dim)
        generated_encoded = torch.softmax(generated_data, dim=-1)  # 转换为概率

        # 解码为SELFIES和SMILES
        selfies_matrix = self.onehot_decode_to_selfies(generated_encoded)
        generated_smiles = []
        for generated_selfies in selfies_matrix:
            mol = Chem.MolFromSmiles(sf.decoder(generated_selfies))
            generated_smiles.append(Chem.MolToSmiles(mol) if mol else "None")

        # 筛选有效SMILES
        return list(set([s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]))

    # 可选：将SMILES转换为带噪输入（生成中未使用，仅供参考）
    def smiles_to_noisy(self, input_smiles_list, t=500):
        input_onehot = self.onehot_encode_from_smiles(input_smiles_list)
        t_tensor = torch.full((len(input_smiles_list),), t, dtype=torch.long)
        noisy_data, _ = self.model.forward_diffusion(input_onehot, t_tensor)
        return noisy_data