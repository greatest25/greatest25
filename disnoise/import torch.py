import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from cnn_trans_model import HybridModel

# 自定义数据集类
"""
数据集类，用于加载和处理图像数据。
"""
class DenoiseDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir=None, transform=None, add_noise=False, noise_level=0.1):
        """
        参数:
            clean_dir (str): 干净图像目录
            noisy_dir (str, optional): 噪声图像目录，如果为None则自动添加噪声
            transform: 图像预处理
            add_noise (bool): 是否自动添加噪声
            noise_level (float): 噪声水平
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.add_noise = add_noise
        self.noise_level = noise_level
        
        # 获取所有图像文件名
        self.image_files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载干净图像
        img_name = self.image_files[idx]
        clean_img_path = os.path.join(self.clean_dir, img_name)
        clean_img = Image.open(clean_img_path).convert('RGB')
        
        if self.transform:
            clean_img = self.transform(clean_img)
        
        # 获取噪声图像
        if self.noisy_dir is not None:
            noisy_img_path = os.path.join(self.noisy_dir, img_name)
            noisy_img = Image.open(noisy_img_path).convert('RGB')
            if self.transform:
                noisy_img = self.transform(noisy_img)
        elif self.add_noise:
            # 自动添加高斯噪声
            noise = torch.randn_like(clean_img) * self.noise_level
            noisy_img = clean_img + noise
        else:
            noisy_img = clean_img
        
        return noisy_img, clean_img

# 训练函数
def train_model(model, train_loader, val_loader=None, num_epochs=50, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}')
        
        # 更新学习率
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for noisy_imgs, clean_imgs in val_loader:
                    noisy_imgs = noisy_imgs.to(device)
                    clean_imgs = clean_imgs.to(device)
                    outputs = model(noisy_imgs)
                    loss = criterion(outputs, clean_imgs)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            print(f'Validation Loss: {val_loss:.6f}')
            scheduler.step(val_loss)
        else:
            scheduler.step(epoch_loss)
    
    return model

# 主函数
if __name__ == "__main__":
    # 设置数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 设置数据集路径
    clean_dir = "e:\\disnoise\\dataset\\GT"  # 修改为您的干净图像目录
    noisy_dir = "e:\\disnoise\\dataset\\noise"   # 修复路径中的转义字符
    
    # 创建数据集
    train_dataset = DenoiseDataset(
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        transform=transform,
        add_noise=False,  # 由于已有噪声图像，设为False
        noise_level=0.1  # 噪声水平
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # 减小批量大小以减少内存使用
        shuffle=True,
        num_workers=2,
        pin_memory=True  # 对GPU训练有帮助
    )
    
    # 初始化模型
    model = HybridModel(in_channels=3, out_channels=3)
    
    # 检查是否有GPU并强制使用GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用设备: {device}")
        print(f"当前使用的GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存使用情况: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    else:
        raise RuntimeError("未检测到GPU，请确保您的系统有NVIDIA GPU并且已正确安装CUDA和PyTorch的CUDA版本")
    
    # 训练模型
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=100,  # 增加训练轮数以获得更好的结果
        lr=0.0005,  # 降低学习率以获得更稳定的训练
        device=device
    )
    
    # 保存模型
    torch.save(trained_model.state_dict(), "e:\\disnoise\\denoiser_model.pth")
    print("模型已保存到 e:\\disnoise\\denoiser_model.pth")