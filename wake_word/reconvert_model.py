#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新转换ONNX模型以修复padding兼容性问题
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime
import numpy as np
from pathlib import Path

class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv1d, self).__init__()
        # 手动计算padding以确保输出长度与输入相同
        # 对于奇数kernel_size，padding = (kernel_size - 1) * dilation // 2
        # 这样可以保持序列长度不变
        padding = (kernel_size - 1) * dilation // 2
        
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

def reconvert_model(model_dir="models/xiaoqi"):
    """重新转换ONNX模型"""
    # 加载配置
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"配置信息: {config}")
    
    # 创建模型
    model = DilatedWakeNet(
        input_dim=config['n_mfcc'], 
        model_dim=config['model_dim']
    )
    
    # 加载训练好的权重
    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 兼容不同的权重格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print("模型权重加载成功")
    else:
        print(f"警告: 找不到权重文件 {checkpoint_path}")
        print("将使用随机初始化的权重")
    
    model.eval()
    
    # 备份原始ONNX文件
    original_onnx_path = os.path.join(model_dir, "wakenet_model.onnx")
    backup_path = os.path.join(model_dir, "wakenet_model_backup.onnx")
    if os.path.exists(original_onnx_path):
        os.rename(original_onnx_path, backup_path)
        print(f"原始ONNX模型已备份为: {backup_path}")
    
    # 创建示例输入
    sample_input = torch.randn(1, config['n_frames'], config['n_mfcc'])
    print(f"示例输入形状: {sample_input.shape}")
    
    # 生成新的ONNX模型
    new_onnx_path = os.path.join(model_dir, "wakenet_model.onnx")
    
    with torch.no_grad():
        # 测试PyTorch模型输出
        pytorch_output = model(sample_input)
        print(f"PyTorch模型输出形状: {pytorch_output.shape}")
        print(f"PyTorch模型输出值: {pytorch_output.numpy()}")
        
        # 导出ONNX
        torch.onnx.export(
            model,
            sample_input,
            new_onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['mfcc'],
            output_names=['logits'],
            dynamic_axes={
                'mfcc': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            verbose=False
        )
    
    print(f"新的ONNX模型已导出: {new_onnx_path}")
    
    # 验证ONNX模型
    try:
        onnx_model = onnx.load(new_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型结构验证通过")
        
        # 测试ONNX Runtime推理
        ort_session = onnxruntime.InferenceSession(new_onnx_path)
        ort_inputs = {'mfcc': sample_input.numpy()}
        ort_outputs = ort_session.run(['logits'], ort_inputs)[0]
        
        print(f"ONNX模型输出形状: {ort_outputs.shape}")
        print(f"ONNX模型输出值: {ort_outputs}")
        
        # 比较PyTorch和ONNX输出
        diff = np.abs(pytorch_output.numpy() - ort_outputs)
        max_diff = np.max(diff)
        print(f"PyTorch vs ONNX 最大差异: {max_diff}")
        
        if max_diff < 1e-5:
            print("✅ ONNX模型输出与PyTorch模型一致")
        else:
            print("⚠️  ONNX模型输出与PyTorch模型有差异")
        
        print("ONNX模型成功转换并验证!")
        return True
        
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
        # 恢复原始文件
        if os.path.exists(backup_path):
            os.rename(backup_path, original_onnx_path)
            print("已恢复原始ONNX文件")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="重新转换ONNX模型")
    parser.add_argument("--model_dir", type=str, default="models/xiaoqi",
                       help="模型目录路径")
    
    args = parser.parse_args()
    
    success = reconvert_model(args.model_dir)
    if success:
        print("模型重新转换成功!")
    else:
        print("模型重新转换失败!") 