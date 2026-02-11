import torch
from torch import nn


def _build_norm(norm_type: str, num_channels: int) -> nn.Module:
    """构建归一化层。"""
    # 兼容大小写输入。  
    norm_key = str(norm_type).lower()
    # 默认 BN，兼容旧权重。  
    if norm_key == "bn":
        return nn.BatchNorm2d(num_channels)
    # IN 在小 batch 下更稳。  
    if norm_key == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    # GN 组数取 32 与通道数的可整除值。  
    if norm_key == "gn":
        group_num = min(32, num_channels)
        while num_channels % group_num != 0 and group_num > 1:
            group_num -= 1
        return nn.GroupNorm(group_num, num_channels)
    # 不使用归一化。  
    if norm_key == "none":
        return nn.Identity()
    # 输入非法时显式报错。  
    raise ValueError(f"Unsupported norm type: {norm_type}")


def _build_activation(act_type: str) -> nn.Module:
    """构建激活函数。"""
    # 兼容大小写输入。  
    act_key = str(act_type).lower()
    # 默认 ReLU，兼容旧实现。  
    if act_key == "relu":
        return nn.ReLU(inplace=True)
    # LeakyReLU 常用于判别器。  
    if act_key == "lrelu":
        return nn.LeakyReLU(0.01, inplace=True)
    # SiLU 在部分最新视觉模型中更平滑。  
    if act_key == "silu":
        return nn.SiLU(inplace=True)
    # 输入非法时显式报错。  
    raise ValueError(f"Unsupported activation type: {act_type}")


class Conv2d(nn.Module):
    """卷积块：Conv + Norm + Act，支持安全残差。"""

    def __init__(
        self,
        cin,
        cout,
        kernel_size,
        stride,
        padding,
        residual=False,
        norm_type="bn",
        act_type="relu",
        use_spectral_norm=False,
        *args,
        **kwargs,
    ):
        # 初始化父类。  
        super().__init__(*args, **kwargs)
        # 构建卷积层。  
        conv = nn.Conv2d(cin, cout, kernel_size, stride, padding)
        # 可选谱归一化，提升 Lipschitz 稳定性。  
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        # 保持旧的模块命名，兼容历史 checkpoint。  
        self.conv_block = nn.Sequential(
            conv,
            _build_norm(norm_type, cout),
        )
        # 构建激活函数。  
        self.act = _build_activation(act_type)
        # 保存残差开关。  
        self.residual = residual

    def forward(self, x):
        """前向传播。"""
        # 主分支卷积。  
        out = self.conv_block(x)
        # 仅在形状一致时做残差，避免静默 shape 错误。  
        if self.residual and out.shape == x.shape:
            out = out + x
        # 返回激活输出。  
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    """无归一化卷积块：Conv + Act。"""

    def __init__(
        self,
        cin,
        cout,
        kernel_size,
        stride,
        padding,
        residual=False,
        act_type="lrelu",
        use_spectral_norm=False,
        *args,
        **kwargs,
    ):
        # 初始化父类。  
        super().__init__(*args, **kwargs)
        # 构建卷积层。  
        conv = nn.Conv2d(cin, cout, kernel_size, stride, padding)
        # 可选谱归一化。  
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        # 保持旧字段名，兼容调用侧。  
        self.conv_block = nn.Sequential(conv)
        # 构建激活。  
        self.act = _build_activation(act_type)

    def forward(self, x):
        """前向传播。"""
        # 执行卷积。  
        out = self.conv_block(x)
        # 返回激活。  
        return self.act(out)


class Conv2dTranspose(nn.Module):
    """反卷积块：Deconv + Norm + Act。"""

    def __init__(
        self,
        cin,
        cout,
        kernel_size,
        stride,
        padding,
        output_padding=0,
        norm_type="bn",
        act_type="relu",
        *args,
        **kwargs,
    ):
        # 初始化父类。  
        super().__init__(*args, **kwargs)
        # 构建反卷积。  
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            _build_norm(norm_type, cout),
        )
        # 构建激活。  
        self.act = _build_activation(act_type)

    def forward(self, x):
        """前向传播。"""
        # 执行反卷积。  
        out = self.conv_block(x)
        # 返回激活。  
        return self.act(out)