import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import DilatedWakeNet
from dataset import WakeWordDataset
from utils import setup_logger, save_config
from convert import convert_to_onnx, quantize_model

def train(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("train")
    logger.info(f"使用设备: {device}")
    
    # 保存配置
    config = {
        "sample_rate": 16000,
        "n_mfcc": 13,
        "window_size_ms": 30,
        "stride_ms": 30,
        "n_frames": 40,
        "model_dim": 32,
        "wake_word": args.wake_word,
        "threshold": 0.5,
        "averaging_frames": 5
    }
    save_config(config, os.path.join(args.output_dir, "config.json"))
    
    # 数据集
    logger.info("加载数据集...")
    train_dataset = WakeWordDataset(
        args.data_dir, 
        wake_word=args.wake_word,
        mode="train", 
        n_mfcc=config["n_mfcc"],
        sample_rate=config["sample_rate"],
        window_size_ms=config["window_size_ms"],
        stride_ms=config["stride_ms"],
        n_frames=config["n_frames"]
    )
    
    val_dataset = WakeWordDataset(
        args.data_dir, 
        wake_word=args.wake_word,
        mode="val",
        n_mfcc=config["n_mfcc"],
        sample_rate=config["sample_rate"],
        window_size_ms=config["window_size_ms"],
        stride_ms=config["stride_ms"],
        n_frames=config["n_frames"]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    
    # 创建模型
    model = DilatedWakeNet(
        input_dim=config["n_mfcc"], 
        model_dim=config["model_dim"]
    ).to(device)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params/1000:.2f}K")
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for mfccs, labels in pbar:
            mfccs = mfccs.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            logits = model(mfccs)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for mfccs, labels in pbar:
                mfccs = mfccs.to(device)
                labels = labels.to(device).float()
                
                logits = model(mfccs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }, os.path.join(args.output_dir, "best_model.pth"))
            logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"加载最佳模型，验证准确率: {checkpoint['val_acc']:.4f}")
    
    # 转换为ONNX和ESP-DL模型
    logger.info("转换为ONNX模型...")
    onnx_path = os.path.join(args.output_dir, "wakenet_model.onnx")
    input_shape = (1, config["n_frames"], config["n_mfcc"])
    convert_to_onnx(model, input_shape, onnx_path)
    
    logger.info("量化模型...")
    esp_model_dir = os.path.join(args.output_dir, "esp_model")
    os.makedirs(esp_model_dir, exist_ok=True)
    quantize_model(onnx_path, esp_model_dir, config)
    
    logger.info(f"训练完成! 模型保存在: {args.output_dir}")
    logger.info(f"ESP-DL兼容模型保存在: {esp_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练轻量级唤醒词模型")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--wake_word", type=str, required=True, help="目标唤醒词")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)
