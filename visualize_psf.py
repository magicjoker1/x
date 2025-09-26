import torch
import matplotlib.pyplot as plt
from data.dataset import Aberration
import os

# 设置中文字体，确保显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def generate_and_visualize_psf(num_samples=3, save_dir="psf_samples"):
    """
    生成并显示PSF图像样本

    参数:
        num_samples: 生成样本数量
        save_dir: 图像保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 初始化Aberration类（参数与配置文件保持一致）
    aberration_gen = Aberration(
        img_size=128,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        precision=torch.float,
        zRange=1.0,
        bias_z=4,
        zernike=[3, 5, 6, 7, 8, 9, 11, 12, 13, 24],  # 常用泽尼克多项式序号        bias_val=[-1,0,1],
        npts=745
    )

    # 生成样本
    for i in range(num_samples):
        # 随机生成泽尼克系数（范围：-1到1）
        num_zernike = len(aberration_gen.zernike)
        C = (torch.rand(num_zernike) * 2 - 1).to(aberration_gen.device)  # 随机系数

        # 生成PSF
        psf, _ = aberration_gen.gen(C)

        # 转换为CPU并可视化
        psf_np = psf.cpu().detach().numpy()

        # 创建子图（每个样本包含3个通道，对应不同bias_val）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"PSF样本 {i + 1} (泽尼克系数: {[f'{c:.2f}' for c in C.cpu().numpy()]})", fontsize=12)

        for j in range(3):
            im = axes[j].imshow(psf_np[j], cmap='viridis')
            axes[j].set_title(f"bias_val = {aberration_gen.bias_val[j]}")
            axes[j].axis('off')
            fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

        # 保存图像
        save_path = os.path.join(save_dir, f"psf_sample_{i + 1}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存PSF样本 {i + 1} 至 {save_path}")

    print(f"\n所有样本已生成，保存目录：{os.path.abspath(save_dir)}")


if __name__ == "__main__":
    generate_and_visualize_psf(num_samples=3)