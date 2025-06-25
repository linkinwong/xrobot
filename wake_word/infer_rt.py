#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时唤醒词检测推理脚本
使用训练好的 ONNX 模型进行实时音频流推理
支持 Linux 和 macOS 跨平台
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

import numpy as np
import librosa
import onnxruntime as ort
import pyaudio
import torch
from scipy.signal import resample


class AudioBuffer:
    """环形音频缓冲区，用于管理音频流数据"""
    
    def __init__(self, max_length_seconds=5, sample_rate=16000):
        """
        初始化音频缓冲区
        
        Args:
            max_length_seconds: 缓冲区最大长度（秒）
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        self.max_length = int(max_length_seconds * sample_rate)
        self.buffer = deque(maxlen=self.max_length)
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data):
        """添加音频数据到缓冲区"""
        with self.lock:
            if isinstance(audio_data, bytes):
                # 转换字节数据为浮点数数组
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
            
            # 添加到缓冲区
            for sample in audio_array:
                self.buffer.append(sample)
    
    def get_audio_segment(self, length_samples):
        """获取指定长度的音频段"""
        with self.lock:
            if len(self.buffer) < length_samples:
                # 如果缓冲区长度不足，用零填充
                audio_segment = np.zeros(length_samples, dtype=np.float32)
                available_samples = len(self.buffer)
                if available_samples > 0:
                    audio_segment[-available_samples:] = list(self.buffer)[-available_samples:]
                return audio_segment
            else:
                # 获取最近的音频数据
                return np.array(list(self.buffer)[-length_samples:], dtype=np.float32)


class MFCCExtractor:
    """MFCC特征提取器"""
    
    def __init__(self, n_mfcc=13, sample_rate=16000, window_size_ms=30, stride_ms=30, n_frames=40):
        """
        初始化MFCC提取器
        
        Args:
            n_mfcc: MFCC特征数量
            sample_rate: 采样率
            window_size_ms: 窗口大小（毫秒）
            stride_ms: 步长（毫秒）
            n_frames: 目标帧数
        """
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.n_frames = n_frames
        
        # 计算窗口和步长（采样点数）
        self.frame_length = int(sample_rate * window_size_ms / 1000)
        self.hop_length = int(sample_rate * stride_ms / 1000)
        
        print(f"MFCC参数: n_mfcc={n_mfcc}, frame_length={self.frame_length}, hop_length={self.hop_length}")
    
    def extract_features(self, audio):
        """
        从音频中提取MFCC特征
        
        Args:
            audio: 音频数组
            
        Returns:
            MFCC特征数组，形状为 (n_frames, n_mfcc)
        """
        try:
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            ).T  # 转置为 (时间帧, 特征)
            
            # 处理特征长度
            if mfcc.shape[0] > self.n_frames:
                # 取最后的帧
                mfcc = mfcc[-self.n_frames:]
            else:
                # 前向填充
                pad_width = self.n_frames - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((pad_width, 0), (0, 0)), mode='constant')
            
            return mfcc.astype(np.float32)
        
        except Exception as e:
            print(f"MFCC特征提取失败: {e}")
            # 返回零特征
            return np.zeros((self.n_frames, self.n_mfcc), dtype=np.float32)


class WakeWordDetector:
    """唤醒词检测器"""
    
    def __init__(self, model_path, config_path, averaging_frames=5):
        """
        初始化唤醒词检测器
        
        Args:
            model_path: ONNX模型路径
            config_path: 配置文件路径
            averaging_frames: 平滑预测的帧数
        """
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
        
        # 加载ONNX模型
        try:
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"成功加载ONNX模型: {model_path}")
            print(f"使用推理提供者: {self.session.get_providers()}")
        except Exception as e:
            print(f"加载ONNX模型失败: {e}")
            raise
        
        # 检测参数
        self.threshold = self.config.get('threshold', 0.5)
        self.averaging_frames = averaging_frames
        self.wake_word = self.config['wake_word']
        
        # 预测历史，用于平滑
        self.prediction_history = deque(maxlen=averaging_frames)
        
        # 统计信息
        self.total_inferences = 0
        self.inference_times = deque(maxlen=100)
        
        print(f"唤醒词检测器初始化完成: {self.wake_word}, 阈值: {self.threshold}")
    
    def predict(self, audio):
        """
        预测音频中是否包含唤醒词
        
        Args:
            audio: 音频数组
            
        Returns:
            tuple: (检测结果, 置信度, 平滑后的置信度)
        """
        start_time = time.time()
        
        try:
            # 提取MFCC特征
            mfcc_features = self.mfcc_extractor.extract_features(audio)
            
            # 准备模型输入
            input_data = np.expand_dims(mfcc_features, axis=0)  # 添加batch维度
            
            # 运行推理
            outputs = self.session.run(None, {'mfcc': input_data})
            logits = outputs[0][0][0]  # 提取logits
            
            # 计算概率
            confidence = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            
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
            print(f"预测失败: {e}")
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


class RealTimeWakeWordDetector:
    """实时唤醒词检测系统"""
    
    def __init__(self, model_path, config_path, device_index=None, 
                 inference_interval_ms=100, averaging_frames=5):
        """
        初始化实时检测系统
        
        Args:
            model_path: ONNX模型路径
            config_path: 配置文件路径
            device_index: 音频设备索引
            inference_interval_ms: 推理间隔（毫秒）
            averaging_frames: 平滑预测的帧数
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config['sample_rate']
        self.inference_interval_ms = inference_interval_ms
        self.device_index = device_index
        
        # 计算音频参数
        self.chunk_size = 1024  # PyAudio块大小
        self.audio_length_seconds = 2.0  # 用于推理的音频长度
        self.audio_length_samples = int(self.audio_length_seconds * self.sample_rate)
        
        # 初始化检测器
        self.detector = WakeWordDetector(model_path, config_path, averaging_frames)
        
        # 初始化音频缓冲区
        self.audio_buffer = AudioBuffer(max_length_seconds=5, sample_rate=self.sample_rate)
        
        # 初始化音频流
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 控制标志
        self.running = False
        self.detection_enabled = True
        
        # 线程
        self.audio_thread = None
        self.inference_thread = None
        
        # 结果队列
        self.result_queue = queue.Queue()
        
        print(f"实时检测系统初始化完成")
        print(f"采样率: {self.sample_rate}, 推理间隔: {inference_interval_ms}ms")
        print(f"音频长度: {self.audio_length_seconds}s ({self.audio_length_samples} 采样点)")
    
    def list_audio_devices(self):
        """列出可用的音频设备"""
        print("可用的音频设备:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print(f"  {i}: {info['name']} - {info['maxInputChannels']} 输入通道")
    
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
        return (None, pyaudio.paContinue)
    
    def _inference_loop(self):
        """推理循环"""
        last_inference_time = 0
        inference_interval_seconds = self.inference_interval_ms / 1000.0
        
        while self.running:
            current_time = time.time()
            
            # 检查是否到了推理时间
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
                        'wake_word': self.detector.wake_word
                    }
                    
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        # 队列满了，丢弃旧结果
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                
                last_inference_time = current_time
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.001)
    
    def start(self):
        """启动实时检测"""
        if self.running:
            print("检测已经在运行中")
            return
        
        print("启动实时唤醒词检测...")
        self.running = True
        
        # 启动音频流
        self.start_audio_stream()
        
        # 启动推理线程
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        print("实时检测启动成功!")
    
    def stop(self):
        """停止实时检测"""
        print("停止实时检测...")
        self.running = False
        
        # 停止音频流
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # 等待线程结束
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
        
        print("实时检测已停止")
    
    def enable_detection(self):
        """启用检测"""
        self.detection_enabled = True
        print("检测已启用")
    
    def disable_detection(self):
        """禁用检测"""
        self.detection_enabled = False
        print("检测已禁用")
    
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
    parser = argparse.ArgumentParser(description="实时唤醒词检测")
    
    # 模型参数
    parser.add_argument("--model_dir", type=str, 
                       default="./output/xiaoqi_20250625_105407",
                       help="模型目录路径")
    
    # 音频参数
    parser.add_argument("--device", type=int, default=None,
                       help="音频设备索引 (使用 --list_devices 查看可用设备)")
    parser.add_argument("--list_devices", action="store_true",
                       help="列出可用的音频设备")
    
    # 推理参数
    parser.add_argument("--inference_interval", type=int, default=100,
                       help="推理间隔(毫秒)")
    parser.add_argument("--averaging_frames", type=int, default=5,
                       help="平滑预测的帧数")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 构建模型文件路径
    model_dir = Path(args.model_dir)
    model_path = model_dir / "wakenet_model.onnx"
    config_path = model_dir / "config.json"
    
    # 检查文件是否存在
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        # 创建检测器
        detector = RealTimeWakeWordDetector(
            model_path=str(model_path),
            config_path=str(config_path),
            device_index=args.device,
            inference_interval_ms=args.inference_interval,
            averaging_frames=args.averaging_frames
        )
        
        # 如果需要列出设备
        if args.list_devices:
            detector.list_audio_devices()
            return
        
        # 启动检测
        detector.start()
        
        print(f"\n开始监听唤醒词: '{detector.detector.wake_word}'")
        print("按 'q' 退出, 'p' 暂停/恢复检测, 's' 显示统计信息")
        print("-" * 60)
        
        # 主循环
        try:
            while True:
                # 获取检测结果
                result = detector.get_result(timeout=0.1)
                
                if result:
                    timestamp_str = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
                    
                    if result['detected']:
                        print(f"🎯 [{timestamp_str}] 检测到唤醒词! "
                              f"置信度: {result['confidence']:.3f}, "
                              f"平滑: {result['smoothed_confidence']:.3f}")
                    elif args.verbose:
                        print(f"   [{timestamp_str}] "
                              f"置信度: {result['confidence']:.3f}, "
                              f"平滑: {result['smoothed_confidence']:.3f}")
                
                # 检查键盘输入
                import select
                import sys
                
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.readline().strip().lower()
                    
                    if key == 'q':
                        break
                    elif key == 'p':
                        if detector.detection_enabled:
                            detector.disable_detection()
                        else:
                            detector.enable_detection()
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
        print("检测已停止")
        
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 