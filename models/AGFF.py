import torch
import torch.nn as nn
import torch.nn.functional as F


class Efficient_Attention_Gate(nn.Module):
    """
    高效注意力门（Efficient Attention Gate, EAG）
    功能：通过分组卷积轻量化设计，生成注意力掩码，引导特征自适应融合
    核心设计：
        - 分组卷积降维：减少计算量，适配高通道特征
        - 门控信号生成：基于两路输入特征的交互，动态生成空间注意力掩码
        - 残差增强：融合后特征与原始特征相加，保留基础信息
    Args:
        F_g: 引导特征（如高层/跨模态特征）通道数
        F_l: 本地特征（如低层特征）通道数
        F_int: 中间交互通道数
        num_groups: 分组卷积组数（默认32）
    """

    def __init__(self, F_g, F_l, F_int, num_groups=2):
        super(Efficient_Attention_Gate, self).__init__()
        self.num_groups = num_groups
        self.grouped_conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )

        self.grouped_conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.psi(self.relu(x1 + g1))
        out = x * psi
        out += x

        return out


class SimpleAttention(nn.Module):
    """
    简易通道注意力模块（Simple Channel Attention）
    功能：基于全局平均池化，生成通道注意力权重，筛选关键通道
    Args:
        in_channels: 输入通道数
        reduction: 通道压缩比例（默认16）
    """

    def __init__(self, in_channels, reduction=16):
        super(SimpleAttention, self).__init__()

        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # 输入的 x 为 (B, C, H, W)
        batch_size, channels, height, width = x.size()
        # 平均池化操作来获取全局上下文信息
        gap = F.adaptive_avg_pool2d(x, (1, 1))  # (B, C, 1, 1)
        gap = gap.view(batch_size, channels)  # (B, C)
        # 通过全连接层得到注意力权重
        attention = F.relu(self.fc1(gap))
        attention = torch.sigmoid(self.fc2(attention))  # (B, C)
        # 将注意力权重应用于输入特征
        attention = attention.view(batch_size, channels, 1, 1)  # (B, C, 1, 1)
        # x = x * attention  # 对特征进行加权

        return attention


class SpatialAttention(nn.Module):
    """
    空间注意力模块：生成空间注意力权重，定位关键空间区域
    Args:
        kernel_size: 卷积核大小（默认7，仅支持3或7）
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DHAR(nn.Module):
    """
    动态混合注意力融合模块（Dynamic Hybrid Attention Fusion, DHAR）
    功能：基于通道注意力权重，动态分配两路特征的融合比例，再经空间注意力增强
    核心设计：
        - 自适应权重分配：根据两路特征的通道重要性，计算动态融合系数
        - 通道-空间注意力协同：先通道加权融合，再空间筛选，双重优化
    Args:
        c1: 第一路输入特征通道数
        c2: 第二路输入特征通道数（输出通道数）
    """

    def __init__(self, c1, c2):
        super(DHAR, self).__init__()
        self.se1 = SimpleAttention(c1)
        self.se2 = SimpleAttention(c2)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.act = nn.Sigmoid()
        self.SA = SpatialAttention(kernel_size=3)  # 替换为膨胀卷积
        self.conv = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )

    # self.transformer = TransformerBlock(dim=c2, num_heads=4)  # 加入 Transformer

    def forward(self, x1, x2):
        weight_x1 = self.se1(x1)
        weight_x2 = self.se2(x2)
        weight_all = self.act(self.GAP(x1 + x2))

        alpha_x1 = weight_x1 / (weight_x1 + weight_x2 + 1e-6)
        alpha_x2 = weight_x2 / (weight_x1 + weight_x2 + 1e-6)
        alpha_x1 = alpha_x1 * weight_all
        alpha_x2 = alpha_x2 * weight_all
        X = alpha_x1 * x1 + alpha_x2 * x2
        X = self.conv(X) * self.SA(X)
        # X =   # 进一步增强特征
        # X = self.conv(X) + self.SA(X)
        return X


class AGFF(nn.Module):
    """
    注意力引导特征融合模块（Attention-Guided Feature Fusion, AGFF）
    功能：整合特征适配、EAG注意力门融合、DHAR动态混合融合，实现跨尺度/跨模态特征精准融合
    核心设计：
        - 双路径融合：EAG侧重空间注意力引导，DHAR侧重动态权重分配，双路径互补
        - 特征适配：1×1卷积统一两路特征通道数，确保融合兼容性
        - 加法融合：双路径输出相加，强化特征互补，避免信息稀释
    Args:
        in_dim_g: 引导特征（如高层/跨模态）输入通道数
        indim_l: 本地特征（如低层）输入通道数（输出通道数）
        is_bottom: 是否为最底层融合（预留参数，当前未使用）
    """

    def __init__(self, in_dim_g, indim_l, is_bottom=False):
        super(AGFF, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim_g, indim_l, kernel_size=1, stride=1),
                                  nn.BatchNorm2d(indim_l),
                                  nn.ReLU())
        self.EAG = Efficient_Attention_Gate(indim_l, indim_l, indim_l)
        self.DHAR = DHAR(indim_l, indim_l)

    def forward(self, x1, x2):
        ##x1_>high  x2->lower
        x1 = self.conv(x1)
        xeag = self.EAG(x1, x2)
        xeag = x1 + xeag
        x_mtr = self.DHAR(x1, x2)
        # X=torch.cat((xeag,x_mtr),dim=1)
        X = xeag + x_mtr
        return X


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    f_d = torch.randn(1, 64, 32, 32).to(device)
    f_s = torch.randn(1, 64, 32, 32).to(device)

    model = AGFF(64, 64).to(device)

    y = model(f_d, f_s)

    print("输入特征维度：", f_d.shape)
    print("输入特征维度：", f_s.shape)
    print("输出特征维度：", y.shape)
    print(torch.__version__)
    print(torch.backends.cudnn.version())