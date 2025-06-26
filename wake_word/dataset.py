import os
import torch
import numpy as np
import librosa
import soundfile as sf
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class WakeWordDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 wake_word="你好小智",
                 mode="train",
                 n_mfcc=13,
                 sample_rate=16000,
                 window_size_ms=30,
                 stride_ms=30,
                 n_frames=40,
                 train_ratio=0.8):
        """
        唤醒词数据集
        
        参数:
            data_dir: 数据目录，应包含 positive/ 和 negative/ 子目录
            wake_word: 目标唤醒词
            mode: 'train' 或 'val'
            n_mfcc: MFCC特征数量
            sample_rate: 音频采样率
            window_size_ms: 窗口大小(毫秒)
            stride_ms: 窗口步长(毫秒)
            n_frames: 帧数
            train_ratio: 训练集比例
        """
        self.data_dir = Path(data_dir)
        self.wake_word = wake_word
        self.mode = mode
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.window_size_ms = window_size_ms
        self.stride_ms = stride_ms
        self.n_frames = n_frames
        
        # 窗口和步长(采样点)
        self.frame_length = int(sample_rate * window_size_ms / 1000)
        self.hop_length = int(sample_rate * stride_ms / 1000)
        
        # 加载文件列表
        self.pos_files = sorted(list((self.data_dir / "positive").glob("*.wav")))
        self.neg_files = sorted(list((self.data_dir / "negative").glob("*.wav")))
        
        # 分割训练集和验证集
        if mode == "train":
            self.pos_files = self.pos_files[:int(len(self.pos_files) * train_ratio)]
            self.neg_files = self.neg_files[:int(len(self.neg_files) * train_ratio)]
        else:
            self.pos_files = self.pos_files[int(len(self.pos_files) * train_ratio):]
            self.neg_files = self.neg_files[int(len(self.neg_files) * train_ratio):]
        
        # 平衡类别
        min_len = min(len(self.pos_files), len(self.neg_files))
        if mode == "train":
            # 训练集重复正样本达到平衡
            repeat_factor = max(1, len(self.neg_files) // len(self.pos_files))
            self.pos_files = self.pos_files * repeat_factor
        
        # 合并文件列表和标签
        self.files = self.pos_files + self.neg_files
        self.labels = [1] * len(self.pos_files) + [0] * len(self.neg_files)
        
        # 打乱数据集
        if mode == "train":
            indices = np.random.permutation(len(self.files))
            self.files = [self.files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # 加载音频文件
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"加载文件失败: {file_path}, {e}")
            # 返回一个空音频作为替代
            audio = np.zeros(self.sample_rate)
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length, 
            hop_length=self.hop_length
        ).T  # (时间帧, 特征)
        
        # 处理特征长度
        if mfcc.shape[0] > self.n_frames:
            # 随机选择起始点，提高模型鲁棒性
            start = np.random.randint(0, mfcc.shape[0] - self.n_frames + 1)
            mfcc = mfcc[start:start+self.n_frames]
        else:
            # 填充
            pad_width = self.n_frames - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        
        return torch.FloatTensor(mfcc), torch.tensor(label)

def create_sample_dataset(output_dir, sample_rate=16000):
    """创建示例数据集，用于测试"""
    os.makedirs(output_dir + "/positive", exist_ok=True)
    os.makedirs(output_dir + "/negative", exist_ok=True)
    
    # 创建正样本 - 正弦波
    for i in range(50):
        t = np.linspace(0, 1, sample_rate)
        signal = np.sin(2*np.pi*440*t) * 0.5
        sf.write(f"{output_dir}/positive/sample_{i}.wav", signal, sample_rate)
    
    # 创建负样本 - 白噪声
    for i in range(200):
        noise = np.random.normal(0, 0.5, sample_rate)
        sf.write(f"{output_dir}/negative/noise_{i}.wav", noise, sample_rate)
