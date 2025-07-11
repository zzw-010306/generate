#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchdiffeq import odeint
import time

# 设备管理
def get_device(gpu_index=None):
    if torch.cuda.is_available():
        if gpu_index is not None and gpu_index < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_index}")
        else:
            device = torch.device("cuda")
        print(f"使用设备: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，回退到: CPU")
    return device

# 向量场定义
class VectorField(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 用于时间维度
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, t, x):
        if t.dim() == 0:  # 如果 t 是标量
            t = t * torch.ones(x.shape[0], device=x.device)
        if t.dim() == 1:  # 如果 t 是 (batch_size,)
            t = t.unsqueeze(1)  # 扩展为 (batch_size, 1)
        if x.dim() == 3:  # 如果 x 是三维 (batch_size, seq_len, feature_dim)
            t = t.unsqueeze(1)  # 扩展为 (batch_size, 1, 1)
            t = t.expand(-1, x.shape[1], 1)  # 扩展为 (batch_size, seq_len, 1)
        elif x.dim() == 2:  # 如果 x 是二维 (batch_size, feature_dim)
            t = t  # 保持 (batch_size, 1)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

class FlowMatchingModel(nn.Module):
    def __init__(self, input_dim = 109):
        super().__init__()
        self.vector_field = VectorField(input_dim=input_dim)
    
    def forward(self, x0, t0=0.0, t1=1.0, steps=10):
        t = torch.linspace(t0, t1, steps, device=x0.device)
        trajectory = odeint(self.vector_field, x0, t)
        return trajectory[-1]
    
    def train_step(self, x1, t0=0.0, t1=1.0):
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device) * (t1 - t0) + t0
        vt_pred = self.vector_field(t, x0)
        vt_target = (x1 - x0) / (t1 - t0)
        loss = nn.MSELoss()(vt_pred, vt_target)
        return loss

if __name__ == "__main__":
    # 初始化设备
    device = get_device(gpu_index=2)
    
    # 加载训练和验证数据
    X_train = np.load("../../data/train_fragment_selfies_onehot.npy")
    X_validation = np.load("../../data/validation_fragment_selfies_onehot.npy")
    
    # 获取特征维度
    feature_dim = X_train.shape[-1]
    print(f"特征维度: {feature_dim}")
    
    trainloader = DataLoader(X_train, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
    validationloader = DataLoader(X_validation, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型和优化器
    model = FlowMatchingModel(input_dim=feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 早停参数
    early_stopping_patience = 30
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_model_path = "../result/flow_matching_model_best.pth"
    
    # 训练循环
    for epoch in range(3000):
        model.train()
        train_loss = 0.0
        for data in trainloader:
            data = data.float().to(device)
            optimizer.zero_grad()
            loss = model.train_step(data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in validationloader:
                data = data.float().to(device)
                val_loss += model.train_step(data).item()
        
        train_loss /= len(trainloader)
        val_loss /= len(validationloader)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型: Val Loss = {best_val_loss:.6f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"早停触发: 在第 {epoch} 轮停止训练")
                break
    
    torch.save(model.state_dict(), "../result/flow_matching_model_final.pth")
    print(f"训练完成，最终模型已保存")