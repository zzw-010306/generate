#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
import selfies as sf
import sys

# 导入 SELFIES 编码器
sys.path.insert( "../GAN")
from SELFIES_onehot_generator import Onehot_Generator

# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN_predict:
    def __init__(self, model_path, label_path, noise_dim=100, hidden_dim=256):
        """初始化 GAN 模型"""
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.label = list(np.array(pd.read_csv(label_path, index_col=0)["Symbol"]))
        self.max_length = 30
        self.input_dim = len(self.label) * self.max_length  # 展平后的输入维度
        self.noise_dim = noise_dim

        # 初始化生成器和判别器
        self.generator = Generator(noise_dim, hidden_dim, self.input_dim).to(self.device)
        self.discriminator = Discriminator(self.input_dim, hidden_dim).to(self.device)

        # 加载预训练模型（如果存在）
        if model_path and torch.cuda.is_available():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator'])
                self.discriminator.load_state_dict(checkpoint['discriminator'])
            except FileNotFoundError:
                print(f"未找到预训练模型 {model_path}，将从头训练")
        self.generator.eval()
        self.discriminator.eval()

    def onehot_encode_from_smiles(self, smiles_list):
        """将 SMILES 转换为独热编码"""
        generator = Onehot_Generator([sf.encoder(smiles) for smiles in smiles_list], 
                                    symbol_label=self.label, max_length=self.max_length)
        onehot_data = generator.onehot(mode="standard")
        return torch.tensor(onehot_data, dtype=torch.float32).to(self.device)

    def onehot_decode_to_selfies(self, encoded_data):
        """将独热编码解码为 SELFIES"""
        restoration_selfies = []
        for i in range(encoded_data.shape[0]):
            indices = torch.argmax(encoded_data[i].reshape(self.max_length, -1), dim=-1).cpu().numpy()
            selfies_str = "".join(np.array(self.label)[indices].tolist()).replace("[_]", "")
            restoration_selfies.append(selfies_str)
        return np.array(restoration_selfies)

    def train_gan(self, real_smiles_list, epochs=100, batch_size=64, lr=0.0002):
        """训练 GAN 模型"""
        self.generator.train()
        self.discriminator.train()
        
        real_data = self.onehot_encode_from_smiles(real_smiles_list)
        dataset = torch.utils.data.TensorDataset(real_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            for real_batch in dataloader:
                real_batch = real_batch[0].view(-1, self.input_dim)  # 展平
                batch_size = real_batch.size(0)

                # 训练判别器
                d_optimizer.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                real_output = self.discriminator(real_batch)
                d_loss_real = criterion(real_output, real_labels)

                z = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(fake_output, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()

                # 训练生成器
                g_optimizer.zero_grad()
                fake_output = self.discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                g_optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # 保存模型
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, 'gan_model.pth')

    def part_generator(self, input_smiles, generate_num=10, random_state=0):
        """使用生成器生成分子片段"""
        np.random.seed(random_state)
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(generate_num, self.noise_dim).to(self.device)
            generated_data = self.generator(z)
            generated_data = generated_data.view(generate_num, self.max_length, -1)
            generated_data = torch.softmax(generated_data, dim=-1)

            selfies_matrix = self.onehot_decode_to_selfies(generated_data)
            generated_smiles = []
            for selfies_str in selfies_matrix:
                mol = Chem.MolFromSmiles(sf.decoder(selfies_str))
                if mol:
                    generated_smiles.append(Chem.MolToSmiles(mol))
                else:
                    generated_smiles.append("None")

        return list(set([s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]))