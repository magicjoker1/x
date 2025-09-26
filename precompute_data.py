import torch
import os
import glob
import time
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data.dataset import NoisyDataset, AberrationDataset


class WrappedDataset(Dataset):
    """简单包装器，确保返回(data, target)格式"""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return item  # 已为(data, target)格式
        return item, torch.tensor([])  # 兼容格式

    def __len__(self):
        return len(self.base)


def precompute_dataset(args, save_path):
    """预生成数据集并保存为.pt文件"""
    os.makedirs(save_path, exist_ok=True)
    print(f"数据保存目录: {save_path}")

    # 检查已有文件
    existing = sorted(glob.glob(os.path.join(save_path, "*.pt")))
    if existing and not args.overwrite:
        print(f"发现 {len(existing)} 个已有文件，跳过生成（使用 --overwrite 强制覆盖）")
        return
    elif existing and args.overwrite:
        print(f"发现 {len(existing)} 个已有文件，将全部覆盖...")

    # 根据数据集类型创建实例
    if args.dataset == "psfNoisy":
        dataset = NoisyDataset(
            dataset_size=args.data_size,
            num_zernike=len(args.zernike),
            val_test_size=int(args.data_size * (args.val_size + args.test_size)),
            zernike=args.zernike,
            bias_val=args.bias_val,
            npts=args.npts,
            z_range=args.z_range,
            precision=torch.float32,
            bias_z=args.bias_z
        )
    elif args.dataset == "psf":
        dataset = AberrationDataset(
            dataset_size=args.data_size,
            num_zernike=len(args.zernike),
            val_test_size=int(args.data_size * (args.val_size + args.test_size)),
            zernike=args.zernike,
            bias_val=args.bias_val,
            npts=args.npts,
            precision=torch.float32,
            bias_z=args.bias_z,
            data_diversity=args.data_diversity
        )
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset}")

    # 包装数据集确保返回(data, target)
    dataset = WrappedDataset(dataset)

    # 使用DataLoader并行生成
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    print(f"开始预生成 {len(dataset)} 个样本（批大小={args.batch_size}, 进程数={args.num_workers}）...")
    start = time.time()

    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Processing batches")):
        # 处理单个样本
        for i in range(data.size(0)):
            idx = batch_idx * args.batch_size + i
            if idx >= len(dataset):
                break  # 防止超出范围
            save_file = os.path.join(save_path, f"{idx:06d}.pt")
            torch.save((data[i], target[i]), save_file)

    elapsed = time.time() - start
    print(f"预生成完成！耗时 {elapsed:.2f}s，保存至 {save_path}")


def main():
    # 直接定义所有需要的参数，避免继承问题
    parser = argparse.ArgumentParser(description="预生成数据集")

    # 基础配置参数
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dataset", type=str, default="psf", choices=["psf", "psfNoisy"], help="数据集类型")
    parser.add_argument("--data_size", type=int, default=10000, help="数据集总大小")
    parser.add_argument("--folder_name", type=str, default="sd10", help="数据保存文件夹名")
    parser.add_argument("--zernike", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        help="泽尼克多项式索引")
    parser.add_argument("--val_size", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--npts", type=int, default=128, help="图像尺寸参数")

    # 预生成专用参数
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的数据")
    parser.add_argument("--batch_size", type=int, default=32, help="预生成批大小")
    parser.add_argument("--num_workers", type=int, default=4, help="预生成进程数")

    # 其他必要参数（从配置文件读取或使用默认值）
    parser.add_argument("--bias_val", type=float, nargs="+", default=[0.0], help="偏差值")
    parser.add_argument("--bias_z", type=int, nargs="+", default=[], help="偏差对应的泽尼克索引")
    parser.add_argument("--z_range", type=float, default=1.0, help="泽尼克系数范围")
    parser.add_argument("--data_diversity", type=float, default=1.0, help="数据多样性参数")

    args = parser.parse_args()
    save_directory = f"./precomputed_data/{args.folder_name}"
    precompute_dataset(args, save_directory)


if __name__ == "__main__":
    main()
