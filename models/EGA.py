import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from typing import List
from torch import Tensor
from mmcv.cnn import build_norm_layer
from math import log
import numpy
import matplotlib.pyplot as plt

try:
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


class Conv_Extra(nn.Module):
    """
    额外特征增强卷积块：1×1降维→3×3特征提取→1×1升维，配合归一化与激活
    功能：对边缘/高斯特征与原始特征的融合结果进行非线性增强，提升特征判别能力
    输入：特征图 [B, C, H, W]（C为输入通道数）
    输出：特征图 [B, C, H, W]（与输入维度一致，确保残差连接兼容）
    """

    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(
            # 1×1卷积：降维至64通道，减少计算量
            nn.Conv2d(channel, 64, 1),
            build_norm_layer(norm_layer, 64)[1],  # 归一化层（如BN）
            act_layer(),  # 激活函数（如ReLU）
            # 3×3卷积：提取局部空间特征，padding=1保持尺寸
            nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, 64)[1],
            act_layer(),
            # 1×1卷积：升维回原通道数，适配后续残差连接
            nn.Conv2d(64, channel, 1),
            build_norm_layer(norm_layer, channel)[1]
        )

    def forward(self, x):
        out = self.block(x)
        return out


class Scharr(nn.Module):
    """
    Scharr边缘检测模块：基于Scharr算子提取方向敏感的边缘特征
    核心优势：1. 对边缘的方向响应更敏感（比Sobel算子更鲁棒）；2. 支持分组卷积，降低计算量
    输入：特征图 [B, C, H, W]
    输出：边缘增强后的特征图 [B, C, H, W]
    """

    def __init__(self, channel, norm_layer, act_layer):
        super(Scharr, self).__init__()
        # 定义Scharr滤波器（x方向：水平边缘，y方向：垂直边缘）
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
                                dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]],
                                dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]

        # 分组卷积：每个通道独立计算边缘，避免通道间干扰（计算量降低C倍）
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        # 将Scharr滤波器权重分配给卷积层（每个通道重复相同滤波器）
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)  # [C,1,3,3]
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)

        self.norm = build_norm_layer(norm_layer, channel)[1]  # 归一化，稳定边缘特征
        self.act = act_layer()  # 激活函数，增强非线性
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)  # 特征增强块

    def forward(self, x):
        # 应用Scharr卷积，提取x/y方向边缘
        edges_x = self.conv_x(x)  # 水平边缘（如遥感图像中的道路横向边缘）
        edges_y = self.conv_y(x)  # 垂直边缘（如建筑物纵向边缘）

        # 边缘融合：平方和开根号，模拟边缘强度（类似Canny边缘检测的梯度幅值计算）
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))  # 归一化+激活，增强边缘判别性

        # 特征融合：原始特征+边缘特征，经增强块输出
        out = self.conv_extra(x + scharr_edge)
        return out


class Gaussian(nn.Module):
    """
    高斯建模模块：基于可学习高斯滤波器，对特征进行不确定性感知的平滑与增强
    核心作用：1. 抑制传感器噪声（如遥感图像的椒盐噪声）；2. 建模特征不确定性，增强鲁棒性
    输入：特征图 [B, C, H, W]
    输出：高斯增强后的特征图 [B, C, H, W]（feature_extra=True时）或纯高斯特征（False时）
    """

    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra  # 是否启用额外特征增强
        # 生成高斯核（固定权重，不参与梯度更新）
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()

        # 分组高斯卷积：每个通道独立平滑，保留通道特异性
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1,
                                  padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)  # [C,1,size,size]

        self.norm = build_norm_layer(norm_layer, dim)[1]  # 归一化，稳定平滑后特征
        self.act = act_layer()  # 激活函数，引入非线性
        # 特征增强块（可选）：增强高斯特征与原始特征的融合能力
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        # 高斯平滑：抑制噪声，保留目标轮廓（如模糊的遥感小目标）
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))  # 归一化+激活，增强平滑特征的判别性

        # 特征融合：原始特征+高斯特征（可选增强）
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        """
        生成2D高斯核：权重符合高斯分布，中心权重高，边缘权重低
        参数：
            size: 核大小（如5）
            sigma: 标准差（控制平滑程度，sigma越大平滑越强）
        返回：高斯核 [1, 1, size, size]
        """
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)  # 扩展为4D张量
        return kernel / kernel.sum()  # 归一化，确保权重和为1，避免特征幅值偏移


class LFEA(nn.Module):
    """
    轻量级特征增强聚合（LFEA）模块：边缘/高斯注意力引导的特征聚合
    核心逻辑：1. 注意力加权边缘/高斯特征；2. 通道注意力筛选关键特征；3. 残差融合保留原始信息
    输入：
        c: 原始特征 [B, C, H, W]
        att: 边缘/高斯注意力特征 [B, C, H, W]
    输出：聚合增强后的特征 [B, C, H, W]
    """

    def __init__(self, channel, norm_layer, act_layer):
        super(LFEA, self).__init__()
        self.channel = channel
        # 计算1D卷积核大小k：基于通道数自适应（log2(channel)+1的一半，确保为奇数）
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1  # 保证k为奇数，方便padding

        # 3×3卷积：增强注意力特征的局部关联性
        self.conv2d = self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化：[B,C,H,W]→[B,C,1,1]
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)  # 通道注意力
        self.sigmoid = nn.Sigmoid()  # 注意力权重归一化
        self.norm = build_norm_layer(norm_layer, channel)[1]  # 最终归一化，稳定输出

    def forward(self, c, att):
        # 注意力融合：原始特征×注意力特征 + 原始特征（增强注意力引导）
        att = c * att + c
        att = self.conv2d(att)  # 增强注意力特征的局部关联性

        # 通道注意力计算：全局池化→1D卷积→Sigmoid
        wei = self.avg_pool(att)  # [B,C,1,1]
        # 维度调整：[B,C,1,1]→[B,1,C]→1D卷积→[B,1,C]→[B,C,1,1]
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)  # 注意力权重（0~1），突出关键通道

        # 特征聚合：原始特征 + 注意力加权特征 → 归一化
        x = self.norm(c + att * wei)
        return x


class LFE_Module(nn.Module):
    """
    LEG核心模块（Lightweight Feature Enhancement Module）：整合边缘检测、高斯建模与特征聚合
    核心流程：边缘/高斯注意力生成 → LFEA特征聚合 → MLP增强 → 残差连接
    输入：特征图 [B, dim, H, W]（dim为输入通道数）
    输出：增强后的特征图 [B, dim, H, W]（与输入维度一致，支持即插即用）
    """

    def __init__(self,
                 dim,  # 输入特征通道数
                 stage,  # 所在网络阶段（0: 边缘检测，≥1: 高斯建模）
                 mlp_ratio,  # MLP通道扩展比例（如2.0表示中间层通道为dim×2）
                 drop_path,  # 随机深度概率（防止过拟合）
                 act_layer,  # 激活函数类型（如nn.ReLU）
                 norm_layer  # 归一化层类型（如dict(type='BN')）
                 ):
        super().__init__()
        self.stage = stage
        # 随机深度模块：训练时随机丢弃部分路径，增强泛化能力
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP模块：增强特征非线性表达（1×1卷积实现，比全连接更高效）
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP中间层通道数
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),  # 升维
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)  # 降维回原通道
        ]
        self.mlp = nn.Sequential(*mlp_layer)

        self.LFEA = LFEA(dim, norm_layer, act_layer)  # 特征聚合模块

        # 根据阶段选择注意力类型：Stage 0用Scharr边缘检测，其他阶段用高斯建模
        if stage == 0:
            self.att_generator = Scharr(dim, norm_layer, act_layer)
        else:
            self.att_generator = Gaussian(dim, 5, 1.0, norm_layer, act_layer)  # 5×5高斯核，sigma=1.0

        self.norm = build_norm_layer(norm_layer, dim)[1]  # MLP输出归一化

    def forward(self, x: Tensor) -> Tensor:
        # 生成注意力特征（边缘或高斯）
        att = self.att_generator(x)
        # LFEA特征聚合：原始特征+注意力特征
        x_att = self.LFEA(x, att)
        # 残差连接：原始特征 + 随机深度(MLP(聚合特征) + 归一化)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)


    '''
                LFE_Module(
                dim=dim,
                stage=stage,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
    '''

    dim = 64
    # stage = (0, 1, 2, 3)
    mlp_ratio = 2.
    drop_path = 0
    # depths = (1, 4, 4, 2)
    depths =1
    act_layer = nn.ReLU
    norm_layer = dict(type='BN', requires_grad=True)

    # drop_path_rate = 0.1
    # dpr = [x.item()
    #        for x in torch.linspace(0, drop_path_rate, sum(depths))]
    # drop_path = dpr[sum(depths[:3]):sum(depths[:3 + 1])]
    # print(drop_path)

    leg = LFE_Module(dim, 0, 2, 0, act_layer, norm_layer).to(device)
    y = leg(x)

    print("微信公众号：十小大的底层视觉工坊")
    print("知乎、CSDN：十小大")
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)