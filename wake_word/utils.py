import os
import json
import logging
from pathlib import Path

def setup_logger(name):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(ch)
    
    return logger

def save_config(config, path):
    """保存配置文件"""
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(path):
    """加载配置文件"""
    with open(path, 'r') as f:
        return json.load(f)

def calculate_model_size(model):
    """计算模型大小(参数量)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def visualize_predictions(model, dataset, num_samples=5):
    """可视化模型预测结果"""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    for i in range(num_samples):
        # 获取样本
        mfcc, label = dataset[i]
        mfcc_batch = mfcc.unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            logits = model(mfcc_batch)
            proba = torch.sigmoid(logits).item()
        
        # 绘制MFCC
        axes[i, 0].imshow(mfcc.numpy().T, aspect='auto', origin='lower')
        axes[i, 0].set_title(f'MFCC - 真实标签: {label.item()}')
        axes[i, 0].set_ylabel('MFCC特征')
        axes[i, 0].set_xlabel('时间帧')
        
        # 绘制预测概率
        axes[i, 1].bar(['非唤醒词', '唤醒词'], [1-proba, proba])
        axes[i, 1].set_title(f'预测概率 - 预测: {proba:.2f} > 0.5 = {proba > 0.5}')
        axes[i, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    return fig
