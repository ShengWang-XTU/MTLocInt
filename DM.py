import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import skimage.transform
from collections import Counter
from sklearn.utils import resample
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


class DiffusionModel(nn.Module):
    def __init__(self, image_channels=3, image_size=256, timesteps=1000, num_labels=4):
        super(DiffusionModel, self).__init__()
        self.image_channels = image_channels
        self.image_size = image_size
        self.timesteps = timesteps
        self.num_labels = num_labels  # 多标签的类别数
        # UNet模型：包括图像和标签信息
        self.unet = UNet(image_channels=self.image_channels, image_size=self.image_size, num_labels=self.num_labels)
        # 生成噪声的调度参数
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))  # noise schedule
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)  # cumulative product of alphas

    def forward(self, x, t, labels=None):
        """执行反向去噪过程，labels为多标签信息"""
        noise = torch.randn_like(x)  # 模拟噪声
        return self.reverse_diffusion(x, t, noise, labels)

    def reverse_diffusion(self, x_t, t, noise, labels=None):
        """根据当前图像x_t进行反向去噪，并且考虑标签"""
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = torch.sqrt(1 - alpha_bar_t)
        A = self.unet(x_t, labels)
        B = torch.sqrt(1 - alpha_bar_t)
        if sigma_t.size(0) == 1:
            C = A * sigma_t
        else:
            C = A.T * sigma_t
        D = noise * B
        E = C + D
        return E

    def q_sample(self, x_0, t, labels=None):
        """前向扩散过程：逐步加入噪声"""
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = torch.sqrt(1 - alpha_bar_t)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0.T + sigma_t * noise.T
        return x_t, noise


class UNet(nn.Module):
    def __init__(self, image_channels, image_size, num_labels):
        super(UNet, self).__init__()
        self.num_labels = num_labels
        self.image_size = image_size
        # 标签嵌入：将标签信息嵌入到网络中
        self.label_embedding = nn.Embedding(num_labels, image_size * image_size)  # 标签嵌入

        # Encoder part (down sampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels + 4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder part (up sampling)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, labels):
        # 将标签嵌入成与图像大小一致的形状，然后拼接到输入图像中
        if x.shape[1] == 3:
            x = x
            label_emb = self.label_embedding(labels.long()).view(1, 4, self.image_size, self.image_size)
        else:
            x = x.T
            label_emb = self.label_embedding(labels.long()).view(labels.size(0), 4, self.image_size, self.image_size)
        x = torch.cat([x, label_emb], dim=1)  # 拼接标签信息

        # UNet的编码解码过程
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def multi_label_loss(predictions, targets):
    """
    使用BCEWithLogitsLoss来处理多标签分类问题
    :param predictions: 网络预测结果，shape (batch_size, num_labels)
    :param targets: 真实标签，shape (batch_size, num_labels)
    """
    return nn.BCEWithLogitsLoss()(predictions, targets)


def train_diffusion_model(model, dataloader, optimizer, epochs=5, timesteps=1000, device="cuda"):
    model.train()
    for epoch in range(epochs):
        num = 0
        epoch_loss = 0
        for inputs, labels in dataloader:
            num += 1
            print(f'DM Epoch: {epoch + 1}, Num: {num}')
            inputs = inputs.to(device)  # 输入图像
            labels = labels.to(device)  # 标签信息，shape (batch_size, num_labels)
            t = torch.randint(0, timesteps, (inputs.size(0),), device=device)  # 随机选择时间步
            # 前向扩散过程
            x_t, noise = model.q_sample(inputs, t, labels)  # x_t是噪声图像，noise是原始噪声
            # 反向去噪过程
            predicted_noise = model(x_t, t, labels)  # 预测去噪的结果
            # 计算损失：预测噪声与实际噪声之间的差异
            loss = multi_label_loss(predicted_noise.T, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")


def generate_augmented_data(model, num_samples, labels, device="cuda"):
    model.eval()
    augmented_images = []
    for _ in range(num_samples):
        noise = torch.randn((1, 3, 256, 256), device=device)  # 随机噪声
        t = torch.tensor([0], device=device)  # 从噪声开始
        generated_image = model(noise, t, labels)  # 通过去噪恢复图像
        augmented_images.append(generated_image.cpu().detach().numpy())
    return np.array(augmented_images)


if __name__ == "__main__":
    start = time.time()
    ddr = 256
    NUM1 = 219
    # Read labels
    dfl = pd.read_csv('label_homo_219.csv')
    dfl_ho = pd.read_csv('label_mus_65.csv')
    lncRNA_loc_labels_o = dfl.values
    lncRNA_loc_labels_ho = dfl_ho.values

    R1 = []
    for c in range(NUM1):
        print(f"{c + 1} lncRNA 219 Rxy")
        p = "CGRxy_3D_RNA_219_" + str(ddr) + "_AC_G/CGRxy_3D_219_" + str(ddr) + "_" + str(c + 1) + ".png"
        transform = transforms.ToTensor()
        img = skimage.io.imread(p)
        img = img[6: 400, 83: 477]
        img = skimage.transform.resize(img, (256, 256))
        img = np.asarray(img, dtype=np.float32)
        imgT = img.T
        R1.append(imgT)
    lncRNA_loc_features_o = np.array(R1)

    # 数据集已经加载为dataloader
    data = TensorDataset(torch.tensor(lncRNA_loc_features_o, dtype=torch.float32),
                         torch.tensor(lncRNA_loc_labels_o, dtype=torch.float32))
    dataloader = DataLoader(data, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化扩散模型
    model = DiffusionModel(image_channels=3, image_size=256, timesteps=1000, num_labels=3).to(device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 训练模型
    train_diffusion_model(model, dataloader, optimizer, epochs=5, timesteps=1000, device=device)

    y = lncRNA_loc_labels_o
    # 统计每个标签组合的频率
    label_counts = Counter([tuple(row) for row in y])
    print("原始标签分布:", label_counts)
    # 找到最频繁的标签组合的数量
    max_count = max(label_counts.values())
    # 对每个标签组合进行重采样
    y_balanced = []
    for label, count in label_counts.items():
        # 找到当前标签组合的所有索引
        indices = [i for i, row in enumerate(y) if tuple(row) == label]
        # 如果当前标签组合的数量少于最大数量，则过采样
        if count < max_count:
            resampled_indices = resample(indices, replace=True, n_samples=max_count, random_state=42)
        # 如果当前标签组合的数量多于最大数量，则欠采样
        elif count > max_count:
            resampled_indices = resample(indices, replace=False, n_samples=max_count, random_state=42)
        # 如果数量正好，则保持不变
        else:
            resampled_indices = indices
        # 将重采样后的标签添加到结果中
        y_balanced.extend(y[resampled_indices])
    # 转换为 numpy 数组
    y_balanced = np.array(y_balanced)
    # 统计均衡后的标签分布
    balanced_label_counts = Counter([tuple(row) for row in y_balanced])
    print("均衡后的标签分布:", balanced_label_counts)
    labels = torch.from_numpy(y_balanced)
    augmented_datas = []
    for label in labels:
        augmented_data = generate_augmented_data(model, num_samples=1, labels=label, device=device)
        augmented_datas.append(np.squeeze(augmented_data))
    augmented_datas = np.array(augmented_datas)

    end = time.time()
    haoshi = end - start
    print(f" haoshi time: {str(haoshi)}")
