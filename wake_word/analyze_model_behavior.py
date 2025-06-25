#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析模型对不同音频类型的响应，验证训练数据问题假设
"""

import os
import sys
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import wave


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
    def __init__(self, input_dim=13, model_dim=32):
        super(DilatedWakeNet, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU()
        )
        
        self.dilated_layers = nn.ModuleList([
            DilatedConv1d(model_dim, model_dim, kernel_size=3, dilation=1),
            DilatedConv1d(model_dim, model_dim, kernel_size=3, dilation=2),
            DilatedConv1d(model_dim, model_dim, kernel_size=3, dilation=4),
            DilatedConv1d(model_dim, model_dim, kernel_size=3, dilation=8),
        ])
        
        self.output_layers = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_dim, 1)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_conv(x)
        
        residual = x
        for layer in self.dilated_layers:
            x = layer(x) + residual
            residual = x
        
        x = self.output_layers(x)
        return x


class AudioAnalyzer:
    def __init__(self, model_path, config_path):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载模型
        self.model = DilatedWakeNet(
            input_dim=self.config['n_mfcc'],
            model_dim=self.config['model_dim']
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"模型加载完成: {model_path}")
    
    def extract_mfcc(self, audio):
        """提取MFCC特征"""
        frame_length = int(self.config['sample_rate'] * self.config['window_size_ms'] / 1000)
        hop_length = int(self.config['sample_rate'] * self.config['stride_ms'] / 1000)
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.config['sample_rate'],
            n_mfcc=self.config['n_mfcc'],
            n_fft=frame_length,
            hop_length=hop_length
        ).T
        
        # 处理长度
        if mfcc.shape[0] > self.config['n_frames']:
            mfcc = mfcc[-self.config['n_frames']:]
        else:
            pad_width = self.config['n_frames'] - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        
        return mfcc.astype(np.float32)
    
    def predict(self, audio):
        """预测置信度"""
        mfcc = self.extract_mfcc(audio)
        input_tensor = torch.FloatTensor(mfcc).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            confidence = torch.sigmoid(logits).item()
        
        return confidence, mfcc
    
    def generate_test_audios(self):
        """生成不同类型的测试音频"""
        duration = 1.2  # 与训练时一致
        sample_rate = self.config['sample_rate']
        samples = int(duration * sample_rate)
        
        test_audios = {}
        
        # 1. 绝对静音
        test_audios['Pure Silence'] = np.zeros(samples, dtype=np.float32)
        
        # 2. 极小白噪音
        test_audios['Tiny Noise'] = np.random.normal(0, 0.001, samples).astype(np.float32)
        
        # 3. 小白噪音
        test_audios['Small Noise'] = np.random.normal(0, 0.01, samples).astype(np.float32)
        
        # 4. 中等白噪音
        test_audios['Medium Noise'] = np.random.normal(0, 0.05, samples).astype(np.float32)
        
        # 5. 大白噪音
        test_audios['Large Noise'] = np.random.normal(0, 0.1, samples).astype(np.float32)
        
        # 6. 模拟TTS静音模式（前后静音，中间有信号）
        tts_like = np.zeros(samples)
        middle_start = samples // 4
        middle_end = 3 * samples // 4
        tts_like[middle_start:middle_end] = np.random.normal(0, 0.1, middle_end - middle_start)
        test_audios['TTS-like'] = tts_like.astype(np.float32)
        
        # 7. 反TTS模式（中间静音，前后有信号）
        anti_tts = np.random.normal(0, 0.05, samples)
        anti_tts[middle_start:middle_end] = 0
        test_audios['Anti-TTS'] = anti_tts.astype(np.float32)
        
        return test_audios
    
    def analyze_mfcc_patterns(self, test_audios):
        """分析MFCC特征模式"""
        results = {}
        
        for name, audio in test_audios.items():
            confidence, mfcc = self.predict(audio)
            
            # 计算MFCC统计信息
            mfcc_mean = np.mean(mfcc)
            mfcc_std = np.std(mfcc)
            mfcc_max = np.max(mfcc)
            mfcc_min = np.min(mfcc)
            
            # 计算零值比例
            zero_ratio = np.sum(mfcc == 0) / mfcc.size
            
            results[name] = {
                'confidence': confidence,
                'mfcc': mfcc,
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'mfcc_max': mfcc_max,
                'mfcc_min': mfcc_min,
                'zero_ratio': zero_ratio,
                'audio_energy': np.mean(audio ** 2)
            }
            
            print(f"{name:12s}: Confidence={confidence:.3f}, MFCC_Mean={mfcc_mean:.3f}, "
                  f"Zero_Ratio={zero_ratio:.3f}, Audio_Energy={results[name]['audio_energy']:.6f}")
        
        return results
    
    def save_visualization(self, results, output_dir="analysis_results"):
        """保存可视化结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 创建置信度对比图
        plt.figure(figsize=(12, 8))
        
        names = list(results.keys())
        confidences = [results[name]['confidence'] for name in names]
        
        # 子图1: 置信度对比
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(names)), confidences)
        plt.xticks(range(len(names)), names, rotation=45)
        plt.ylabel('Confidence')
        plt.title('Confidence by Audio Type')
        plt.ylim(0, 1)
        
        # 为置信度>0.5的添加颜色标记
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            if conf > 0.5:
                bar.set_color('red')
                plt.text(i, conf + 0.02, f'{conf:.3f}', ha='center', fontweight='bold')
            else:
                bar.set_color('blue')
                plt.text(i, conf + 0.02, f'{conf:.3f}', ha='center')
        
        # 子图2: MFCC零值比例
        plt.subplot(2, 2, 2)
        zero_ratios = [results[name]['zero_ratio'] for name in names]
        plt.bar(range(len(names)), zero_ratios)
        plt.xticks(range(len(names)), names, rotation=45)
        plt.ylabel('MFCC Zero Ratio')
        plt.title('Zero Values in MFCC Features')
        
        # 子图3: 音频能量 vs 置信度
        plt.subplot(2, 2, 3)
        energies = [results[name]['audio_energy'] for name in names]
        plt.scatter(energies, confidences)
        for i, name in enumerate(names):
            plt.annotate(name, (energies[i], confidences[i]), fontsize=8)
        plt.xlabel('Audio Energy')
        plt.ylabel('Confidence')
        plt.title('Audio Energy vs Confidence')
        
        # 子图4: MFCC均值 vs 置信度
        plt.subplot(2, 2, 4)
        mfcc_means = [results[name]['mfcc_mean'] for name in names]
        plt.scatter(mfcc_means, confidences)
        for i, name in enumerate(names):
            plt.annotate(name, (mfcc_means[i], confidences[i]), fontsize=8)
        plt.xlabel('MFCC Mean')
        plt.ylabel('Confidence')
        plt.title('MFCC Mean vs Confidence')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'audio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存详细的MFCC特征图
        self.plot_mfcc_details(results, output_dir)
    
    def plot_mfcc_details(self, results, output_dir):
        """绘制详细的MFCC特征"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (name, data) in enumerate(results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            mfcc = data['mfcc']
            
            # 绘制MFCC特征热图
            im = ax.imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'{name}\nConfidence: {data["confidence"]:.3f}')
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('MFCC Coefficients')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mfcc_features.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_real_recordings(self, recording_dir="audio_recordings"):
        """分析真实录音"""
        recording_dir = Path(recording_dir)
        if not recording_dir.exists():
            print(f"录音目录不存在: {recording_dir}")
            return
        
        print("\n=== 真实录音分析 ===")
        
        wav_files = list(recording_dir.glob("*.wav"))
        if not wav_files:
            print("未找到录音文件")
            return
        
        for wav_file in sorted(wav_files)[:5]:  # 分析前5个文件
            try:
                # 加载音频
                audio, sr = librosa.load(wav_file, sr=self.config['sample_rate'])
                
                # 计算音频统计信息
                audio_energy = np.mean(audio ** 2)
                audio_max = np.max(np.abs(audio))
                
                # 预测
                confidence, mfcc = self.predict(audio)
                
                print(f"{wav_file.name}: Confidence={confidence:.3f}, "
                      f"Energy={audio_energy:.6f}, Max_Amplitude={audio_max:.3f}")
                
            except Exception as e:
                print(f"处理 {wav_file} 时出错: {e}")


def main():
    model_dir = Path("models/xiaoqi")
    model_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.json"
    
    if not model_path.exists() or not config_path.exists():
        print(f"模型文件不存在: {model_path} 或 {config_path}")
        return
    
    # 创建分析器
    analyzer = AudioAnalyzer(str(model_path), str(config_path))
    
    print("=== 生成测试音频 ===")
    test_audios = analyzer.generate_test_audios()
    
    print("\n=== 分析MFCC模式 ===")
    results = analyzer.analyze_mfcc_patterns(test_audios)
    
    print("\n=== 保存可视化结果 ===")
    analyzer.save_visualization(results)
    
    # 分析真实录音
    analyzer.analyze_real_recordings()
    
    print("\n=== 分析结论 ===")
    # 检查是否存在"静音高置信度"问题
    silence_confidence = results['Pure Silence']['confidence']
    small_noise_confidence = results['Tiny Noise']['confidence']
    
    if silence_confidence > 0.5 or small_noise_confidence > 0.5:
        print("❌ 发现问题: 模型对静音/极小噪音有较高置信度")
        print("   这说明模型可能学到了错误的特征关联")
        print("   建议解决方案:")
        print("   1. 为正例数据添加背景噪音")
        print("   2. 增加更多的静音负例样本")
        print("   3. 重新训练模型")
    else:
        print("✅ 模型对静音的响应正常")


if __name__ == "__main__":
    main() 