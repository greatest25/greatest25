import torch
from torchvision import transforms
from PIL import Image
from cnn_trans_model import HybridModel
import os
import numpy as np

# 添加评估函数
def calculate_psnr(img1, img2):
    """计算峰值信噪比(PSNR)，值越高表示图像质量越好"""
    img1 = np.array(img1).astype(np.float32) / 255.0
    img2 = np.array(img2).astype(np.float32) / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def denoise_image(model, image_path, output_path=None, device='cuda'):
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    # 保存原始尺寸
    orig_size = img.size
    
    # 调整图像大小为256x256，与训练时一致
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        denoised_img = model(img_tensor)
    
    # 转换回PIL图像
    denoised_img = denoised_img.squeeze(0).cpu()
    denoised_img = torch.clamp(denoised_img, 0, 1)
    denoised_img = transforms.ToPILImage()(denoised_img)
    
    # 调整回原始尺寸
    denoised_img = denoised_img.resize(orig_size, Image.LANCZOS)
    
    # 计算去噪前后的图像质量差异
    psnr_value = calculate_psnr(img, denoised_img)
    print(f"图像 {os.path.basename(image_path)} 的PSNR值: {psnr_value:.2f}dB")
    
    # 保存结果
    if output_path:
        denoised_img.save(output_path)
        print(f"已保存去噪图像到: {output_path}")
    
    # 清理GPU内存
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return denoised_img, psnr_value

def process_test_folder(model, input_folder, output_folder, device='cuda'):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有PNG图像
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    print(f"找到 {len(image_files)} 张图像需要处理")
    
    # 处理每张图像
    total_psnr = 0
    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, f"denoised_{img_file}")
        
        print(f"处理图像: {img_file}")
        _, psnr = denoise_image(model, input_path, output_path, device)
        total_psnr += psnr
    
    # 计算平均PSNR
    avg_psnr = total_psnr / len(image_files) if image_files else 0
    print(f"\n所有图像的平均PSNR: {avg_psnr:.2f}dB")
    print("PSNR值越高表示去噪效果越好，通常超过30dB表示良好的去噪效果")

if __name__ == "__main__":
    # 加载模型
    model = HybridModel()
    model.load_state_dict(torch.load("e:\\disnoise\\denoiser_model.pth"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 设置输入和输出文件夹
    input_folder = "e:\\disnoise\\data"  # 包含0000x.png格式图像的文件夹
    output_folder = "e:\\disnoise\\denoised_results"  # 去噪结果保存的文件夹
    
    # 处理整个文件夹
    process_test_folder(model, input_folder, output_folder, device)