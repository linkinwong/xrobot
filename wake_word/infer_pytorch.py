#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用原始PyTorch模型进行推理，验证torch vs onnx的差异
"""

import os
import sys
import argparse
import json
import time
import threading
import queue
from collections import deque
from pathlib import Path
import wave

import numpy as np
import librosa
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample


class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding="same"):
        super(DilatedConv1d, self).__init__()
        # 手动计算padding以确保输出长度与输入相同
        # padding = (kernel_size - 1) * dilation // 2
        
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


class AudioBuffer:
    """环形音频缓冲区，用于管理音频流数据"""
    
    def __init__(self, max_length_seconds=5, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_length = int(max_length_seconds * sample_rate)
        self.buffer = deque(maxlen=self.max_length)
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data):
        """添加音频数据到缓冲区"""
        with self.lock:
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
            
            for sample in audio_array:
                self.buffer.append(sample)
    
    def get_audio_segment(self, length_samples):
        """获取指定长度的音频段"""
        with self.lock:
            if len(self.buffer) < length_samples:
                audio_segment = np.zeros(length_samples, dtype=np.float32)
                available_samples = len(self.buffer)
                if available_samples > 0:
                    audio_segment[-available_samples:] = list(self.buffer)[-available_samples:]
                return audio_segment
            else:
                return np.array(list(self.buffer)[-length_samples:], dtype=np.float32)


class MFCCExtractor:
    """MFCC特征提取器"""
    
    def __init__(self, n_mfcc=13, sample_rate=16000, window_size_ms=30, stride_ms=30, n_frames=40):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.n_frames = n_frames
        
        self.frame_length = int(sample_rate * window_size_ms / 1000)
        self.hop_length = int(sample_rate * stride_ms / 1000)
        
        print(f"MFCC参数: n_mfcc={n_mfcc}, frame_length={self.frame_length}, hop_length={self.hop_length}")
    
    def extract_features(self, audio):
        """提取MFCC特征"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            ).T  # 转置为 (时间帧, 特征)
            
            # 处理特征长度 - 与训练时保持一致
            if mfcc.shape[0] > self.n_frames:
                mfcc = mfcc[-self.n_frames:]
            else:
                # 与训练时保持一致：后向填充（在末尾填充）
                pad_width = self.n_frames - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            
            return mfcc.astype(np.float32)
        
        except Exception as e:
            print(f"MFCC特征提取失败: {e}")
            return np.zeros((self.n_frames, self.n_mfcc), dtype=np.float32)


class PyTorchWakeWordDetector:
    """PyTorch唤醒词检测器"""
    
    def __init__(self, model_path, config_path, averaging_frames=3):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"加载配置: {self.config}")
        
        # 初始化MFCC提取器
        self.mfcc_extractor = MFCCExtractor(
            n_mfcc=self.config['n_mfcc'],
            sample_rate=self.config['sample_rate'],
            window_size_ms=self.config['window_size_ms'],
            stride_ms=self.config['stride_ms'],
            n_frames=self.config['n_frames']
        )
        
        # 创建并加载PyTorch模型
        self.model = DilatedWakeNet(
            input_dim=self.config['n_mfcc'],
            model_dim=self.config['model_dim']
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"成功加载PyTorch模型: {model_path}")
        
        # 检测参数
        self.threshold = 0.8  # 使用高阈值
        self.averaging_frames = averaging_frames
        self.wake_word = self.config['wake_word']
        
        # 预测历史，用于平滑
        self.prediction_history = deque(maxlen=averaging_frames)
        
        # 统计信息
        self.total_inferences = 0
        self.inference_times = deque(maxlen=100)
        
        print(f"PyTorch唤醒词检测器初始化完成: {self.wake_word}, 阈值: {self.threshold}")
    
    def predict(self, audio):
        """预测音频中是否包含唤醒词"""
        start_time = time.time()
        
        try:
            # 提取MFCC特征
            mfcc_features = self.mfcc_extractor.extract_features(audio)
            
            # 准备模型输入
            input_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0)  # 添加batch维度
            
            # 运行推理
            with torch.no_grad():
                logits = self.model(input_tensor)
                logits_value = logits.item()
            
            # 计算概率
            confidence = 1.0 / (1.0 + np.exp(-logits_value))  # sigmoid
            
            # 添加到历史记录
            self.prediction_history.append(confidence)
            
            # 计算平滑后的置信度
            smoothed_confidence = np.mean(list(self.prediction_history))
            
            # 判断是否检测到唤醒词
            detected = smoothed_confidence > self.threshold
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_inferences += 1
            
            return detected, confidence, smoothed_confidence
            
        except Exception as e:
            print(f"PyTorch预测失败: {e}")
            return False, 0.0, 0.0
    
    def get_stats(self):
        """获取推理统计信息"""
        if len(self.inference_times) == 0:
            return {
                'total_inferences': self.total_inferences,
                'avg_inference_time': 0.0,
                'max_inference_time': 0.0,
                'min_inference_time': 0.0
            }
        
        return {
            'total_inferences': self.total_inferences,
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times)
        }


class RealTimePyTorchDetector:
    """实时PyTorch唤醒词检测系统"""
    
    def __init__(self, model_path, config_path, device_index=None, 
                 inference_interval_ms=300, averaging_frames=3, save_audio=False):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config['sample_rate']
        self.inference_interval_ms = inference_interval_ms
        self.device_index = device_index
        self.save_audio = save_audio
        
        # 计算音频参数 - 与训练时保持一致
        expected_duration = (self.config['n_frames'] * self.config['stride_ms']) / 1000.0
        self.audio_length_seconds = expected_duration
        self.audio_length_samples = int(self.audio_length_seconds * self.sample_rate)
        
        self.chunk_size = 1024
        
        # 初始化检测器
        self.detector = PyTorchWakeWordDetector(model_path, config_path, averaging_frames)
        
        # 初始化音频缓冲区
        self.audio_buffer = AudioBuffer(max_length_seconds=5, sample_rate=self.sample_rate)
        
        # 音频录制相关
        if self.save_audio:
            self.audio_save_dir = Path("audio_recordings")
            self.audio_save_dir.mkdir(exist_ok=True)
            self.recording_buffer = deque(maxlen=int(2.0 * self.sample_rate))  # 2秒的音频
            self.last_save_time = 0
            self.save_interval = 2.0  # 每2秒保存一次
            self.recording_counter = 0
            print(f"音频录制已启用，保存目录: {self.audio_save_dir}")
        
        # 初始化音频流
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 控制标志
        self.running = False
        self.detection_enabled = True
        
        # 线程
        self.inference_thread = None
        
        # 结果队列
        self.result_queue = queue.Queue()
        
        print(f"PyTorch实时检测系统初始化完成")
        print(f"采样率: {self.sample_rate}, 推理间隔: {inference_interval_ms}ms")
        print(f"音频长度: {self.audio_length_seconds:.1f}s ({self.audio_length_samples} 采样点)")
        print(f"期望帧数: {self.config['n_frames']}, 每帧: {self.config['stride_ms']}ms")
    
    def start_audio_stream(self):
        """启动音频流"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            print(f"音频流启动成功, 设备: {self.device_index or '默认'}")
        except Exception as e:
            print(f"启动音频流失败: {e}")
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if self.running:
            self.audio_buffer.add_audio(in_data)
            
            # 如果启用了音频录制，将数据添加到录制缓冲区
            if self.save_audio:
                audio_array = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                for sample in audio_array:
                    self.recording_buffer.append(sample)
                    
        return (None, pyaudio.paContinue)
    
    def save_audio_segment(self, audio_data, filename):
        """保存音频片段到文件"""
        try:
            # 转换为16位整数
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # 保存为WAV文件
            with wave.open(str(filename), 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
                
            print(f"✅ 保存音频: {filename}")
        except Exception as e:
            print(f"❌ 保存音频失败: {e}")
    
    def _inference_loop(self):
        """推理循环"""
        last_inference_time = 0
        inference_interval_seconds = self.inference_interval_ms / 1000.0
        
        while self.running:
            current_time = time.time()
            
            # 检查是否需要保存音频
            if self.save_audio and current_time - self.last_save_time >= self.save_interval:
                if len(self.recording_buffer) >= int(1.0 * self.sample_rate):  # 至少1秒的音频
                    # 获取最近2秒的音频数据
                    audio_segment = np.array(list(self.recording_buffer), dtype=np.float32)
                    
                    # 生成文件名
                    timestamp = time.strftime("%H%M%S", time.localtime(current_time))
                    filename = self.audio_save_dir / f"recording_{timestamp}_{self.recording_counter:03d}.wav"
                    
                    # 保存音频
                    self.save_audio_segment(audio_segment, filename)
                    
                    self.recording_counter += 1
                    self.last_save_time = current_time
            
            if current_time - last_inference_time >= inference_interval_seconds:
                if self.detection_enabled:
                    # 获取音频数据
                    audio_data = self.audio_buffer.get_audio_segment(self.audio_length_samples)
                    
                    # 运行推理
                    detected, confidence, smoothed_confidence = self.detector.predict(audio_data)
                    
                    # 放入结果队列
                    result = {
                        'timestamp': current_time,
                        'detected': detected,
                        'confidence': confidence,
                        'smoothed_confidence': smoothed_confidence,
                        'wake_word': self.detector.wake_word,
                        'model_type': 'PyTorch'
                    }
                    
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                
                last_inference_time = current_time
            
            time.sleep(0.001)
    
    def start(self):
        """启动实时检测"""
        if self.running:
            print("检测已经在运行中")
            return
        
        print("启动PyTorch实时唤醒词检测...")
        self.running = True
        
        # 启动音频流
        self.start_audio_stream()
        
        # 启动推理线程
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        print("PyTorch实时检测启动成功!")
    
    def stop(self):
        """停止实时检测"""
        print("停止PyTorch实时检测...")
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
        
        print("PyTorch实时检测已停止")
    
    def get_result(self, timeout=None):
        """获取检测结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self):
        """获取统计信息"""
        return self.detector.get_stats()
    
    def __del__(self):
        """析构函数"""
        self.stop()
        if hasattr(self, 'audio'):
            self.audio.terminate()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PyTorch实时唤醒词检测")
    
    parser.add_argument("--model_dir", type=str, 
                       default="./models/xiaoqi",
                       help="模型目录路径")
    parser.add_argument("--device", type=int, default=None,
                       help="音频设备索引")
    parser.add_argument("--inference_interval", type=int, default=300,
                       help="推理间隔(毫秒)")
    parser.add_argument("--averaging_frames", type=int, default=3,
                       help="平滑预测的帧数")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    parser.add_argument("--save_audio", action="store_true",
                       help="每2秒保存音频片段用于验证")
    
    args = parser.parse_args()
    
    # 构建模型文件路径
    model_dir = Path(args.model_dir)
    model_path = model_dir / "best_model.pth"  # 使用PyTorch权重文件
    config_path = model_dir / "config.json"
    
    # 检查文件是否存在
    if not model_path.exists():
        print(f"错误: PyTorch模型文件不存在: {model_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        # 创建检测器
        detector = RealTimePyTorchDetector(
            model_path=str(model_path),
            config_path=str(config_path),
            device_index=args.device,
            inference_interval_ms=args.inference_interval,
            averaging_frames=args.averaging_frames,
            save_audio=args.save_audio
        )
        
        # 启动检测
        detector.start()
        
        print(f"\n开始监听唤醒词 (PyTorch版本): '{detector.detector.wake_word}'")
        print("按 'q' 退出, 's' 显示统计信息")
        print("-" * 60)
        
        # 主循环
        try:
            while True:
                result = detector.get_result(timeout=0.1)
                
                if result:
                    timestamp_str = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
                    
                    if result['detected']:
                        print(f"🎯 [{timestamp_str}] [PyTorch] 检测到唤醒词! "
                              f"置信度: {result['confidence']:.3f}, "
                              f"平滑: {result['smoothed_confidence']:.3f}")
                    elif args.verbose:
                        print(f"   [{timestamp_str}] [PyTorch] "
                              f"置信度: {result['confidence']:.3f}, "
                              f"平滑: {result['smoothed_confidence']:.3f}")
                
                # 检查键盘输入
                import select
                import sys
                
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.readline().strip().lower()
                    
                    if key == 'q':
                        break
                    elif key == 's':
                        stats = detector.get_stats()
                        print(f"\n统计信息:")
                        print(f"  总推理次数: {stats['total_inferences']}")
                        print(f"  平均推理时间: {stats['avg_inference_time']*1000:.2f}ms")
                        print(f"  最大推理时间: {stats['max_inference_time']*1000:.2f}ms")
                        print(f"  最小推理时间: {stats['min_inference_time']*1000:.2f}ms")
                        print()
        
        except KeyboardInterrupt:
            pass
        
        # 停止检测
        detector.stop()
        print("PyTorch检测已停止")
        
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 