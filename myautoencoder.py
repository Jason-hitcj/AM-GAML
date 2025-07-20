import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 修复设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. 加载和预处理数据
data = pd.read_csv("data/preprocessed_data_ml.csv")
data = data.drop('target', axis=1)
X = data.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 3. 转换为 PyTorch Tensor 并移至设备
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)

# 4. 定义 Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 5. 初始化模型并移至设备
input_dim = X_train.shape[1]
latent_dim = 8
model = Autoencoder(input_dim, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
min_val_loss = float('inf')
patience = 10
counter = 0

# 6. 训练循环
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, X_train)
    loss.backward()
    optimizer.step()
    
    # 验证损失
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, X_test)
    scheduler.step(val_loss)
    
    # 早停逻辑
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# 7. 测试
with torch.no_grad():
    reconstructed = model(X_test)
    test_loss = criterion(reconstructed, X_test)
    print(f"Test MSE: {test_loss:.4f}")

    # 异常检测示例：计算每个样本的 MSE
    mse_per_sample = torch.mean((X_test - reconstructed) ** 2, dim=1)
    print("Top 5 anomalous samples:", mse_per_sample.topk(5).indices)


# import matplotlib.pyplot as plt

# idx = 3218  # 最高异常样本索引
# original = X_test[idx].cpu().numpy()
# reconstructed = reconstructed[idx].cpu().numpy()

# plt.figure(figsize=(10, 4))
# plt.plot(original, label="Original")
# plt.plot(reconstructed, label="Reconstructed")
# plt.legend()
# plt.title(f"Anomaly Sample (MSE: {mse_per_sample[idx]:.4f})")
# # 保存文件（支持PNG/PDF/SVG等格式）
# plt.savefig("img/autoencoder.png", dpi=300, bbox_inches="tight")  # bbox_inches防止截断

# # 主动关闭图形，避免内存占用
# plt.close()  # 关键！阻止图形显示
