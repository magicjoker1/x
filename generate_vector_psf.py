import torch
import matplotlib.pyplot as plt
import os
from data.dataset import Aberration  # 导入现有Aberration类
from utils.seed import seed_worker  # 导入种子设置工具
from utils.summary import get_outdir  # 用于创建输出目录

def generate_vector_psf_plots():
    # 配置参数（参考config中的sd10设置）
    config = {
        "img_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "precision": torch.float32,
        "bias_z": 4,
        "zernike": [3, 5, 6, 7, 8, 9, 11, 12, 13, 24],  # 从sd10配置选取
        "bias_val": [-1, 0, 1],  # 三个偏置值对应三张图
        "npts": 745,  # 从配置文件选取
        "seed": 42,  # 固定种子确保可复现
        "output_folder": "vector_psf_plots"  # 存储图片的文件夹名称
    }

    # 设置随机种子
    seed_worker(config["seed"])
    torch.manual_seed(config["seed"])
    if config["device"] == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

    # 创建输出文件夹（使用项目中已有的get_outdir工具函数）
    output_dir = get_outdir("./results", config["output_folder"])
    print(f"PSF图片将保存至: {output_dir}")

    # 初始化像差生成器（假设已支持矢量光束）
    aberration_generator = Aberration(
        img_size=config["img_size"],
        device=config["device"],
        precision=config["precision"],
        bias_z=config["bias_z"],
        zernike=config["zernike"],
        bias_val=config["bias_val"],
        npts=config["npts"]
    )

    # 生成随机泽尼克系数（控制像差）
    num_zernike = len(config["zernike"])
    C = torch.randn(num_zernike, device=config["device"]) * 0.3  # 像差强度

    # 生成矢量光束PSF
    psf_stack, _ = aberration_generator.gen(C)  # shape: [3, 128, 128]

    # 可视化并保存结果
    plt.figure(figsize=(18, 6))
    for i in range(len(config["bias_val"])):
        plt.subplot(1, 3, i + 1)
        psf_np = psf_stack[i].cpu().detach().numpy()
        plt.imshow(psf_np, cmap="inferno")
        plt.title(f"Vector PSF (Bias: {config['bias_val'][i]})", fontsize=12)
        plt.colorbar(label="Normalized Intensity")
        plt.axis("off")

        # 单独保存每张图片
        img_path = os.path.join(output_dir, f"vector_psf_bias_{config['bias_val'][i]}.png")
        plt.imsave(img_path, psf_np, cmap="inferno")
        print(f"已保存单张图片: {img_path}")

    # 保存组合图
    combined_path = os.path.join(output_dir, "vector_psf_combined.png")
    plt.tight_layout()
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    print(f"已保存组合图片: {combined_path}")
    plt.show()

if __name__ == "__main__":
    generate_vector_psf_plots()