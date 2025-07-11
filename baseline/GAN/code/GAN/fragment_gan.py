import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
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

# 生成器模型（不变）
class Generator(nn.Module):
    def __init__(self, noise_dim, max_length, feature_dim, hidden_dim=128):
        super().__init__()
        self.noise_dim = noise_dim
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(noise_dim, max_length * hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, z):
        batch_size = z.size(0)
        h = self.fc1(z)  # (batch_size, max_length * hidden_dim)
        h = h.view(batch_size, self.max_length, self.hidden_dim)  # (batch_size, max_length, hidden_dim)
        h = h.permute(0, 2, 1)  # (batch_size, hidden_dim, max_length)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = h.permute(0, 2, 1)  # (batch_size, max_length, hidden_dim)
        output = self.fc2(h)  # (batch_size, max_length, feature_dim)
        return output

    def generate(self, batch_size, device):
        z = torch.randn(batch_size, self.noise_dim, device=device)
        with torch.no_grad():
            output = self(z)
        return torch.softmax(output, dim=-1)

# 判别器模型（修改：移除sigmoid，返回线性输出）
class Discriminator(nn.Module):
    def __init__(self, max_length, feature_dim, hidden_dim=128):
        super().__init__()
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, feature_dim, max_length)
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = self.pool(h).squeeze(-1)  # (batch_size, hidden_dim)
        output = self.fc(h)  # (batch_size, 1) 线性输出，无sigmoid
        return output

def evaluate_generator(generator, validation_data, noise_dim, device):
    # 生成样本
    z = torch.randn(validation_data.size(0), noise_dim, device=device)
    with torch.no_grad():
        generated_samples = generator(z)
    # 计算评估指标（这里用欧氏距离作为示例，实际可用 FID 等）
    distance = torch.mean(torch.norm(generated_samples - validation_data, dim=(1, 2)))
    return distance.item()

# 主训练循环
def main():
    # 使用相对路径
    base_dir = "/root"
    os.makedirs(base_dir, exist_ok=True)
    model_path = os.path.join(base_dir, "fragment_gan_generator.pth")
    history_path = os.path.join(base_dir, "history_fragment_gan.csv")
    time_path = os.path.join(base_dir, "calculation_time_fragment_gan.txt")

    # 加载数据
    try:
        X_train = np.load("../data/train_fragment_selfies_onehot.npy")
        X_validation = np.load("../data/validation_fragment_selfies_onehot.npy")
    except Exception as e:
        print(f"数据文件加载失败: {e}")
        return

    # 数据加载器
    trainloader = torch.utils.data.DataLoader(X_train, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型和优化器
    device = get_device()
    max_length = X_train.shape[1]
    feature_dim = X_train.shape[2]
    noise_dim = 100
    generator = Generator(noise_dim, max_length, feature_dim).to(device)
    discriminator = Discriminator(max_length, feature_dim).to(device)
    optimizer_G = optim.RMSprop(generator.parameters(), lr=0.0001)  # WGAN推荐RMSprop
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.0001)

    # 训练参数
    epochs = 3000
    history = []
    eval_interval = 10  # 每10个epoch评估一次
    best_eval_score = float('inf')  # 用于保存最佳模型
    patience = 30  # 早停耐心值：30个epoch没有改善则停止
    counter = 0  # 计数器：记录自上次改善以来经过的epoch数
    clip_value = 0.01  # 权重裁剪值

    # 将验证集移到GPU
    validation_data = try_gpu(torch.tensor(X_validation).float(), device)

    # 训练循环
    st = time.time()
    for epoch in range(epochs):
        for data in trainloader:
            data = try_gpu(data.float(), device)
            batch_size = data.size(0)

            # 训练判别器（更新5次）
            for _ in range(5):
                optimizer_D.zero_grad()
                output_real = discriminator(data)
                z = torch.randn(batch_size, noise_dim, device=device)
                fake_data = generator(z).detach()
                output_fake = discriminator(fake_data)
                loss_D = -torch.mean(output_real) + torch.mean(output_fake)  # WGAN Loss D（注意负号，因为优化器最小化）
                loss_D.backward()
                optimizer_D.step()

                # 权重裁剪
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # 训练生成器（更新1次）
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(z)
            output_fake = discriminator(fake_data)
            loss_G = -torch.mean(output_fake)  # WGAN Loss G
            loss_G.backward()
            optimizer_G.step()

        # 记录损失
        history.append([loss_D.item(), loss_G.item()])
        pd.DataFrame(history, columns=["loss_D", "loss_G"]).to_csv(history_path, index=False)

        # 打印损失
        print(f"Epoch {epoch+1}/{epochs}, Loss D: {loss_D.item():.6f}, Loss G: {loss_G.item():.6f}")

        # 定期评估生成器
        if (epoch + 1) % eval_interval == 0:
            eval_score = evaluate_generator(generator, validation_data, noise_dim, device)
            print(f"Epoch {epoch+1}, Evaluation Score: {eval_score:.6f}")
            # 检查是否改善
            if eval_score < best_eval_score:
                best_eval_score = eval_score
                torch.save(generator.state_dict(), model_path)
                print(f"生成器模型已更新并保存到: {model_path}")
                counter = 0  # 重置计数器
            else:
                counter += eval_interval  # 累加间隔的epoch数
            # 检查早停条件
            if counter >= patience:
                print(f"早停触发：已连续 {counter} 个epoch 验证分数没有改善。")
                break

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