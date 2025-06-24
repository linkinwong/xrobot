import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding="same"):
        super(DilatedConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            dilation=dilation, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class DilatedWakeNet(nn.Module):
    """
    轻量级的基于空洞卷积的唤醒词检测模型，专为ESP32-S3设计
    """
    def __init__(self, input_dim=13, model_dim=32):
        super(DilatedWakeNet, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        # 输入层
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU()
        )
        
        # 空洞卷积层
        self.dilated_layers = nn.ModuleList([
            DilatedConv1d(model_dim, model_dim, kernel_size=3, dilation=1),
            DilatedConv1d(model_dim, model_dim, kernel_size=3, dilation=2),
            DilatedConv1d(model_dim, model_dim, kernel_size=3, dilation=4),
            DilatedConv1d(model_dim, model_dim, kernel_size=3, dilation=8),
        ])
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_dim, 1)
        )
    
    def forward(self, x):
        """
        输入: (batch_size, n_frames, n_mfcc)
        输出: (batch_size, 1) - 唤醒词检测得分
        """
        # 转置为卷积格式 (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch_size, n_mfcc, n_frames)
        
        # 输入卷积
        x = self.input_conv(x)
        
        # 空洞卷积层，带有残差连接
        residual = x
        for layer in self.dilated_layers:
            x = layer(x) + residual
            residual = x
        
        # 输出层
        x = self.output_layers(x)
        
        return x  # 返回logits
