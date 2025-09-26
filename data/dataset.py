from PIL import Image
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
import random
import math
import torch.nn.functional as F
from pathlib import Path
import os


class Aberration():
    def __init__(self, img_size, device, precision=torch.half, zRange=1.0, bias_z=4, zernike=[3, 5, 6, 7],
                 bias_val=[-1, 0, 1], npts=97):
        self.zRange_start = 0
        self.zRange_end = 200  # 0-200 ->-1.0 to 1.0
        if zRange == 0.5:
            self.zRange_start = 50
            self.zRange_end = 150
        self.zRange = zRange
        self.tRange = int(zRange * 100)
        self.img_size = img_size
        self.device = device

        self.precisionFloat = precision
        self.precisionInt = torch.int
        if self.precisionFloat == torch.half:
            self.precisionInt = torch.short
        self.bias_z = bias_z

        self.num_channel = len(bias_val)
        self.bias_val = bias_val
        self.zernike = zernike
        self.npts = npts  # 125
        self.npad = 2001  # 2001
        self.nhpad = math.ceil(self.npad / 2) - 1
        self.nrange = 64
        self.nex = int(round(self.npad - self.npts) / 2)

        self.znorm = 0
        self.phaseapp = self.znorm * math.pi / 2
        self.z_sub = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]
        self.defocus = [0.0]

        self.obs = [self.npts / self.npad, 1, 1]
        self.pup = self.annulus()
        self.circ_res = self.circ(self.npts)
        self.rad = self.circ_res * self.circrad(self.npts)
        self.rad = F.pad(self.rad, (self.nex, self.nex, self.nex, self.nex))
        self.pupPhase = self.pup * torch.exp(1j * self.phaseapp * self.rad * self.rad)
        # 极坐标计算（rho: 径向距离，theta: 角向角度）
        x = torch.linspace(-1, 1, steps=self.npts, device=self.device, dtype=self.precisionFloat)
        x = x.unsqueeze(1).repeat(1, self.npts)  # 形状 [npts, npts]
        y = x.t()  # y为x的转置
        self.rho = torch.sqrt(x ** 2 + y ** 2)  # 径向坐标
        self.theta = torch.atan2(y, x)  # 角向坐标（-π到π）

        # 径向振幅分布 E_rho
        self.E_rho = self.rho * torch.exp(-2 * self.rho ** 2)
        self.zernike_cartesian_matrix()

    def annulus(self):
        if self.obs[2] > 1:
            print('error outer ring out of aperture')
            return

        mattemp = self.circrad(self.npad)

        pup = torch.zeros(self.npad, self.npad, device=self.device, dtype=self.precisionFloat)
        temp1 = self.obs[0] * torch.ones(self.npad, self.npad, device=self.device, dtype=self.precisionFloat)
        temp2 = self.obs[1] * torch.ones(self.npad, self.npad, device=self.device, dtype=self.precisionFloat)
        temp3 = self.obs[2] * torch.ones(self.npad, self.npad, device=self.device, dtype=self.precisionFloat)
        pup = (mattemp <= temp1) | ((mattemp > temp2) & (mattemp <= temp3))

        nh = math.ceil(self.npad / 2) - 1
        if self.obs[0] == 0:
            pup[nh, nh] = 0  # 遮挡中心像素

        return pup

    def circrad(self, npts):
        xtemp = torch.linspace(-1, 1, steps=npts, device=self.device, dtype=self.precisionFloat)
        ytemp = xtemp

        mattemp1 = xtemp.view(len(xtemp), 1) @ torch.ones(1, npts, device=self.device, dtype=self.precisionFloat)
        mattemp2 = 1j * torch.ones(npts, 1, device=self.device) * ytemp
        mattemp = mattemp1 + mattemp2
        mattemp = abs(mattemp)

        return mattemp

    def circ(self, npts):
        mattemp = self.circrad(npts)
        return (mattemp <= torch.ones(npts, npts, device=self.device))

    def zernike_cartesian_matrix(self):
        # 简化版泽尼克多项式函数
        x = torch.linspace(-1, 1, steps=self.npts, device=self.device, dtype=self.precisionFloat)  # 单位半径
        x = torch.flip(x, dims=(0,))
        x = x.repeat(self.npts, 1)
        y = x.t()

        Z = torch.zeros(28, self.npts, self.npts, dtype=self.precisionFloat, device=self.device)
        Z[0, :, :] = 1 * torch.ones(self.npts, self.npts, dtype=self.precisionFloat, device=self.device)
        Z[1, :, :] = 2 * x  # 倾斜
        Z[2, :, :] = 2 * y
        Z[3, :, :] = math.sqrt(6) * (2 * x * y)  # 像散
        Z[4, :, :] = math.sqrt(3) * ((2 * x * x) + (2 * y * y) - 1)  # 离焦
        Z[5, :, :] = math.sqrt(6) * ((-x * x) + (y * y))  # 像散
        Z[6, :, :] = 2 * math.sqrt(2) * ((-x ** 3) + (3 * x * y * y))  # 三叶形像差
        Z[7, :, :] = 2 * math.sqrt(2) * ((-2 * x) + (3 * x ** 3) + (3 * x * y * y))  # 彗差
        Z[8, :, :] = 2 * math.sqrt(2) * ((-2 * y) + (3 * y ** 3) + (3 * y * x * x))  # 彗差
        Z[9, :, :] = 2 * math.sqrt(2) * ((-y ** 3) + (3 * y * x * x))  # 三叶形像差
        Z[10, :, :] = math.sqrt(10) * ((-4 * x ** 3 * y) + (4 * x * y ** 3))
        Z[11, :, :] = math.sqrt(10) * ((-6 * x * y) + (8 * y * x ** 3) + (8 * x * y ** 3))
        Z[12, :, :] = math.sqrt(5) * (
                    1 - (6 * x * x) - (6 * y * y) + (6 * x ** 4) + (12 * x * x * y * y) + (6 * y ** 4))
        Z[13, :, :] = math.sqrt(10) * ((3 * x * x) - (3 * y * y) - (4 * x ** 4) + (4 * y ** 4))
        Z[14, :, :] = math.sqrt(10) * ((x ** 4) - (6 * y ** 2 * x ** 2) + (y ** 4))
        Z[15, :, :] = 2 * math.sqrt(3) * ((x ** 5) - (10 * x ** 3 * y ** 2) + (5 * x * y ** 4))
        Z[16, :, :] = 2 * math.sqrt(3) * (
                    (4 * x ** 3) - (12 * y ** 2 * x) - (5 * x ** 5) + (10 * y ** 2 * x ** 3) + (15 * y ** 4 * x))
        Z[17, :, :] = 2 * math.sqrt(3) * (
                    (3 * x) - (12 * x ** 3) - (12 * x * y ** 2) + (10 * x ** 5) + (20 * x ** 3 * y ** 2) + (
                        10 * x * y ** 4))
        Z[18, :, :] = 2 * math.sqrt(3) * (
                    (3 * y) - (12 * y ** 3) - (12 * y * x ** 2) + (10 * y ** 5) + (20 * x ** 2 * y ** 3) + (
                        10 * y * x ** 4))
        Z[19, :, :] = 2 * math.sqrt(3) * (
                    -(4 * y ** 3) + (12 * x ** 2 * y) + (5 * y ** 5) - (10 * x ** 2 * y ** 3) - (15 * x ** 4 * y))
        Z[20, :, :] = 2 * math.sqrt(3) * ((y ** 5) - (10 * y ** 3 * x ** 2) + (5 * y * x ** 4))
        Z[21, :, :] = math.sqrt(14) * ((6 * x ** 5 * y) - (20 * x ** 3 * y ** 3) + (6 * x * y ** 5))
        Z[22, :, :] = math.sqrt(14) * ((20 * x ** 3 * y) - (20 * x * y ** 3) - (24 * x ** 5 * y) + (24 * x * y ** 5))
        Z[23, :, :] = math.sqrt(14) * ((12 * x * y) - (40 * x ** 3 * y) - (40 * x * y ** 3) + (30 * x ** 5 * y) + (
                    60 * x ** 3 * y ** 3) + (30 * x * y ** 5))
        Z[24, :, :] = math.sqrt(7) * (
                    -1 + (12 * x * x) + (12 * y * y) - (30 * x ** 4) - (60 * x ** 2 * y ** 2) - (30 * y ** 4) + (
                        20 * x ** 6) + (60 * x ** 4 * y ** 2) + (60 * x ** 2 * y ** 4) + (20 * y ** 6))
        Z[25, :, :] = math.sqrt(14) * (-(6 * x ** 2) + (6 * y ** 2) + (20 * x ** 4) - (20 * y ** 4) - (15 * x ** 6) - (
                    15 * x ** 4 * y ** 2) + (15 * x ** 2 * y ** 4) + (15 * y ** 6))
        Z[26, :, :] = math.sqrt(14) * (
                    -(5 * x ** 4) + (30 * x ** 2 * y ** 2) - (5 * y ** 4) + (6 * x ** 6) - (30 * x ** 4 * y ** 2) - (
                        30 * y ** 4 * x ** 2) + (6 * y ** 6))
        Z[27, :, :] = math.sqrt(14) * (-(x ** 6) + (15 * x ** 4 * y ** 2) - (15 * y ** 4 * x ** 2) + (y ** 6))

        self.Z = Z * self.circ_res

    def gen(self, C=None):
        aberration = torch.empty(size=(self.num_channel, self.img_size, self.img_size), device=self.device)

        # 1. 计算泽尼克多项式总和（像差相位）
        Zsum_noDef = torch.zeros(self.npts, self.npts, device=self.device, dtype=self.precisionFloat)
        for j, m1 in enumerate(self.zernike):
            if C[j] != 0:
                Zsum_noDef += C[j] * self.Z[m1, :, :]

        for j, k in enumerate(self.bias_val):
            # 2. 叠加偏置项（如离焦）后的总相位
            if k != 0:
                Zsum = Zsum_noDef + k * self.Z[self.bias_z, :, :]
            else:
                Zsum = Zsum_noDef

            # 3. 生成像差相位因子（exp(1j * 相位)）
            phase = torch.exp(1j * Zsum)

            # 4. 分离径向偏振光的x/y分量，并应用相位
            Ex = self.E_rho * torch.cos(self.theta) * phase  # x分量
            Ey = self.E_rho * torch.sin(self.theta) * phase  # y分量

            # 5. 对分量进行填充（与原代码保持一致的尺寸扩展）
            Ex_pad = F.pad(Ex, (self.nex, self.nex, self.nex, self.nex))
            Ey_pad = F.pad(Ey, (self.nex, self.nex, self.nex, self.nex))

            # 6. 应用光瞳函数（原有物理相位）
            Ex_pupil = self.pupPhase * Ex_pad
            Ey_pupil = self.pupPhase * Ey_pad

            # 7. 分别计算x/y分量的傅里叶变换（得到PSF）
            def compute_psf(component):
                component_fft = torch.fft.ifftshift(component)
                component_fft = torch.fft.fft2(component_fft)
                component_fft = torch.fft.fftshift(component_fft)
                return abs(component_fft) ** 2  # 振幅平方为强度

            psf_x = compute_psf(Ex_pupil)
            psf_y = compute_psf(Ey_pupil)
            psf_total = psf_x + psf_y  # 总PSF为两个分量之和

            # 8. 裁剪到目标尺寸并归一化，增加数值稳定性处理
            outsmall = psf_total[self.nhpad - (self.nrange - 1):self.nhpad + (self.nrange + 1),
                       self.nhpad - (self.nrange - 1):self.nhpad + (self.nrange + 1)]

            # 修复：添加epsilon防止除零，并确保非负
            max_val = torch.max(outsmall)
            eps = torch.finfo(outsmall.dtype).eps  # 获取最小正数，防止除零
            if max_val < eps:
                outsmall = torch.zeros_like(outsmall)
            else:
                outsmall = outsmall / max_val
                # 确保归一化后没有负值（由于浮点误差）
                outsmall = torch.clamp(outsmall, min=0.0)

            aberration[j, :, :] = outsmall

        return aberration, C


def gen_gaussian_kernel(kernel_size=128, sigma=10):
    # 创建1D高斯核
    kernel_1d = torch.tensor(
        [torch.exp(-(x - kernel_size // 2) ** 2 / torch.tensor(2 * sigma ** 2)) for x in range(kernel_size)],
        device='cuda')
    gaussian_kernel = torch.outer(kernel_1d, kernel_1d)
    return gaussian_kernel


def add_poisson_noise(image, noise_level):
    """向图像添加泊松噪声，确保输入非负"""
    # 关键修复：强制确保输入非负，避免Poisson噪声生成错误
    image_clamped = torch.clamp(image, min=0.0)
    noisy_image = torch.poisson(image_clamped * noise_level) / noise_level
    return noisy_image


def add_gaussian_noise(image, mean=0, std=0.01):
    """向图像添加高斯噪声，确保结果在合理范围内"""
    # 确保噪声在与图像相同的设备上生成
    device = image.device
    noise = torch.randn(image.size(), device=device) * std + mean
    # 限制结果在[0, 1]范围内
    noisy_image = torch.clamp(image + noise, min=0.0, max=1.0)
    return noisy_image


class AberrationStrehl(Aberration):
    def __init__(self, img_size, device, precision=torch.half, zRange=1.0, bias_z=4, zernike=[3, 5, 6, 7],
                 bias_val=[-1, 0, 1], npts=97):
        super().__init__(img_size, device, precision, zRange, bias_z, zernike, bias_val, npts)

    def gen(self, C=None):
        aberration = torch.empty(size=(self.num_channel, self.img_size, self.img_size), device=self.device)
        Zsum_noDef = torch.zeros(self.npts, self.npts, device=self.device, dtype=self.precisionFloat)
        for j, m1 in enumerate(self.zernike):
            if C[j] != 0:
                Zsum_noDef += C[j] * self.Z[m1, :, :]

        for j, k in enumerate(self.bias_val):
            Zsum = Zsum_noDef + (k * self.Z[self.bias_z, :, :]) if k != 0 else Zsum_noDef
            phase = torch.exp(1j * Zsum)

            # 分离x/y分量并计算PSF
            Ex = self.E_rho * torch.cos(self.theta) * phase
            Ey = self.E_rho * torch.sin(self.theta) * phase

            Ex_pad = F.pad(Ex, (self.nex, self.nex, self.nex, self.nex))
            Ey_pad = F.pad(Ey, (self.nex, self.nex, self.nex, self.nex))

            Ex_pupil = self.pupPhase * Ex_pad
            Ey_pupil = self.pupPhase * Ey_pad

            # 计算PSF（与父类逻辑一致）
            def compute_psf(component):
                component_fft = torch.fft.ifftshift(component)
                component_fft = torch.fft.fft2(component_fft)
                component_fft = torch.fft.fftshift(component_fft)
                return abs(component_fft) ** 2

            psf_x = compute_psf(Ex_pupil)
            psf_y = compute_psf(Ey_pupil)
            psf_total = psf_x + psf_y

            outsmall = psf_total[self.nhpad - (self.nrange - 1):self.nhpad + (self.nrange + 1),
                       self.nhpad - (self.nrange - 1):self.nhpad + (self.nrange + 1)]

            # 同样添加数值稳定性处理
            max_val = torch.max(outsmall)
            eps = torch.finfo(outsmall.dtype).eps
            if max_val < eps:
                outsmall = torch.zeros_like(outsmall)
            else:
                outsmall = outsmall / max_val
                outsmall = torch.clamp(outsmall, min=0.0)

            aberration[j, :, :] = outsmall  # Strehl类直接使用强度

        return aberration, C


class AberrationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_size, num_zernike, precision, bias_z, val_test_size, zernike, bias_val, npts):
        self.dataset_size = dataset_size
        self.zRange_start = 0
        self.zRange_end = 200  # 0-200 ->-1.0 to 1.0
        self.val_test_size = val_test_size
        self.num_zernike = num_zernike
        self.C_val_test = torch.randint(self.zRange_start, self.zRange_end, size=(val_test_size, self.num_zernike))
        self.C_val_test = (self.C_val_test - 100) * 0.01

        self.gen_aberration = Aberration(128, device='cuda', precision=precision, bias_z=bias_z, zernike=zernike,
                                         bias_val=bias_val, npts=npts)

    def __getitem__(self, idx):
        if idx >= self.val_test_size:
            gen_tr = True
            while gen_tr:
                C = torch.randint(self.zRange_start, self.zRange_end, size=(self.num_zernike,))
                C = (C - 100) * 0.01
                is_equal = torch.eq(self.C_val_test, C).all(dim=1)
                # 确保训练集与验证集系数不重复
                if not is_equal.any():
                    gen_tr = False
            aberration, coeffs = self.gen_aberration.gen(C=C)
        else:
            aberration, coeffs = self.gen_aberration.gen(C=self.C_val_test[idx])

        return aberration, coeffs

    def __len__(self):
        return self.dataset_size


class NoisyDataset(torch.utils.data.Dataset):
    """带噪声的数据集"""

    def __init__(self, dataset_size, num_zernike, precision, bias_z, val_test_size, zernike, bias_val, npts, z_range):
        self.dataset_size = dataset_size
        self.zRange_start = 0
        self.zRange_end = 200  # 0-200 ->-1.0 to 1.0
        if z_range != 1.0:
            self.zRange_start = int((1.0 - z_range) * 100)
            self.zRange_end = int(100 + (z_range * 100))
        C_val_test = torch.randint(self.zRange_start, self.zRange_end, size=(val_test_size, num_zernike))
        self.C_val_test = (C_val_test - 100) * 0.01
        self.val_test_size = val_test_size
        self.num_zernike = num_zernike
        self.channel_len = len(bias_val)
        self.gen_aberration = Aberration(128, device='cuda', precision=precision, bias_z=bias_z, zernike=zernike,
                                         bias_val=bias_val, npts=npts)
        self.gaussian_kernel = gen_gaussian_kernel(kernel_size=128, sigma=30)  # previous sigma is 20

    def __getitem__(self, idx):
        C = torch.zeros(self.num_zernike, )
        if idx >= self.val_test_size:
            gen_tr = True
            while gen_tr:
                C = torch.randint(self.zRange_start, self.zRange_end, size=(self.num_zernike,))
                C = (C - 100) * 0.01
                is_equal = torch.eq(self.C_val_test, C).all(dim=1)
                # 确保训练集与验证集系数不重复
                if not is_equal.any():
                    gen_tr = False
        else:
            C = self.C_val_test[idx]

        aberration, coeffs = self.gen_aberration.gen(C=C)

        for i in range(self.channel_len):
            aberration[i] = aberration[i] * self.gaussian_kernel

            # 修复：增强归一化的数值稳定性
            max_val = aberration[i].max()
            eps = torch.finfo(aberration[i].dtype).eps
            if max_val < eps:
                aberration[i] = torch.zeros_like(aberration[i])
            else:
                aberration[i] /= max_val

            # 添加噪声前再次确保非负
            aberration[i] = torch.clamp(aberration[i], min=0.0)
            aberration[i] = add_poisson_noise(aberration[i], noise_level=2000)
            aberration[i] = add_gaussian_noise(aberration[i], mean=0, std=0.02)

        return aberration, C

    def __len__(self):
        return self.dataset_size


def generate_array(num_zernike):
    array = torch.rand(num_zernike) * 0.4 - 0.2
    zeros = (array == 0).sum().item()

    # 确保至少有5个零元素
    while zeros < 5:
        index = torch.randint(0, num_zernike, (1,))
        array[index] = 0
        zeros = (array == 0).sum().item()

    return array


class PSF_RealDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, exts=['jpg', 'jpeg', 'png', 'PNG'], img_size=128, channel=[0, 1, 2]):
        import os
        # 打印关键信息用于调试
        print("当前工作目录:", os.getcwd())
        print("配置文件中的路径:", data_path)
        print("实际拼接的路径:", os.path.join(os.getcwd(), data_path))
        print("文件是否存在:", os.path.isfile(data_path))
        # 确认无误后再读取
        self.ds = pd.read_csv(data_path)
        self.dir = data_path.rsplit('/', 1)[0]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
        ])
        self.img_size = img_size
        self.channel = channel

    def __getitem__(self, idx):
        d = self.ds.iloc[idx]
        coeffs = torch.tensor(d[1:].tolist())
        aberration = torch.zeros(len(self.channel), self.img_size, self.img_size)
        for i, j in enumerate(self.channel):
            im = Image.open(self.dir + '/' + str(int(d[0])) + '_' + str(j) + '.png')
            red, _, _ = im.split()  # 红色通道包含最多信息
            red = self.transform(red)
            red = torch.from_numpy(np.asarray(red))
            # 确保归一化正确
            max_val = red.max()
            if max_val > 0:
                aberration[i, :, :] = red / max_val
            else:
                aberration[i, :, :] = torch.zeros_like(red)
        return aberration, coeffs

    def __len__(self):
        return len(self.ds)