import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import time
import os

# 设备管理函数
def get_device(gpu_index=2):
    if torch.cuda.is_available() and torch.cuda.device_count() > gpu_index:
        device = torch.device(f"cuda:{gpu_index}")
        print(f"使用设备: {device}")
    else:
        device = torch.device("cpu")
        print(f"cuda:{gpu_index} 不可用，回退到: {device}")
    return device

def try_gpu(tensor, device):
    return tensor.to(device)

# U-Net 模型
class UNet(nn.Module):
    def __init__(self, input_dim=109, hidden_dim=128, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.down1 = nn.LSTM(input_dim + time_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.down2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.down_conv = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bottleneck = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.up1 = nn.LSTM(hidden_dim * 4, hidden_dim, batch_first=True, bidirectional=True)
        self.up2 = nn.LSTM(hidden_dim * 4, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, input_dim)
        self.skip_conv1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.skip_conv2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        if x.dtype != torch.float32:
            x = x.float()
        if t.dtype != torch.float32:
            t = t.unsqueeze(-1).float()
        t_embed = self.pos_encoding(t, self.time_dim)
        t_embed = self.time_embed(t_embed)
        t_embed = t_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, t_embed], dim=-1)
        x1, _ = self.down1(x)
        x2, _ = self.down2(x1)
        x2 = self.down_conv(x2)
        x3, _ = self.bottleneck(x2)
        x2_adjusted = self.skip_conv1(x2)
        x = self.up1(torch.cat([x3, x2_adjusted], dim=-1))[0]
        x1_adjusted = self.skip_conv2(x1)
        x = self.up2(torch.cat([x, x1_adjusted], dim=-1))[0]
        return self.out(x)

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=1000, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = unet.to(self.device)
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)

    def forward_diffusion(self, x0, t, noise):
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1).to(self.device)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt

    def forward(self, x, t):
        return self.unet(x, t)

    def sample(self, batch_size, max_length, feature_dim):
        x = torch.randn(batch_size, max_length, feature_dim, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            noise_pred = self.forward(x, t_tensor)
            alpha_t = self.alphas[t].to(self.device)
            alpha_bar_t = self.alpha_bars[t].to(self.device)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            if t > 0:
                x += torch.sqrt(self.betas[t]) * torch.randn_like(x, device=self.device)
        return torch.softmax(x, dim=-1)

# 损失函数
def diffusion_loss(model, x0):
    batch_size = x0.size(0)
    device = x0.device
    t = torch.randint(0, model.timesteps, (batch_size,), device=device)
    noise = torch.randn_like(x0, device=device)
    xt = model.forward_diffusion(x0, t, noise)
    noise_pred = model(xt, t)
    return nn.MSELoss()(noise_pred, noise)

# 训练和验证函数
def train_epoch(model, trainloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for data in trainloader:
        data = try_gpu(data.float(), device)
        optimizer.zero_grad()
        loss = diffusion_loss(model, data)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    return running_loss / len(trainloader)

def validate_epoch(model, validationloader, device):
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for data in validationloader:
            data = try_gpu(data.float(), device)
            val_loss = diffusion_loss(model, data)
            val_running_loss += val_loss.item()
    return val_running_loss / len(validationloader)

# 主训练循环
def main():
    # 使用相对路径（假设运行时在项目根目录）
    base_dir = "/root"
    os.makedirs(base_dir, exist_ok=True)
    model_path = os.path.join(base_dir, "fragment_diffusion.pth")
    history_path = os.path.join(base_dir, "history_fragment_diffusion.csv")
    time_path = os.path.join(base_dir, "calculation_time_fragment_diffusion.txt")

    # 加载数据
    try:
        X_train = np.load("train_fragment_selfies_onehot.npy")
        X_validation = np.load("validation_fragment_selfies_onehot.npy")
    except Exception as e:
        print(f"数据文件加载失败: {e}")
        return

    # 数据加载器
    trainloader = torch.utils.data.DataLoader(X_train, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    validationloader = torch.utils.data.DataLoader(X_validation, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型和优化器
    device = get_device()
    model = DiffusionModel(UNet(), device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 训练参数
    es_num = 30
    es_count = 0
    epochs = 3000
    min_loss = float('inf')
    history = []

    # 训练循环
    st = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, trainloader, optimizer, device)
        val_loss = validate_epoch(model, validationloader, device)
        
        history.append([train_loss, val_loss])
        pd.DataFrame(history, columns=["loss", "val_loss"]).to_csv(history_path, index=False)

        # 早停逻辑
        if val_loss < min_loss:
            min_loss = val_loss
            es_count = 0  # 重置计数器
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存: 验证损失 {val_loss:.6f}")
        else:
            es_count += 1
            if es_count >= es_num:
                print(f"早停触发: 连续 {es_num} 次验证损失未降低")
                break

        scheduler.step(val_loss)  # 调整学习率
        print(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")

        # 每10轮释放显存
        if (epoch + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存训练时间
    et = time.time()
    with open(time_path, "a") as writer:
        writer.write(f"{et - st}\n")
    print(f"训练完成，总耗时: {et - st:.2f} 秒")

if __name__ == "__main__":
    main()