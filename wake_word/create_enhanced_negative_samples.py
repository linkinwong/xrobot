import numpy as np
import soundfile as sf
from pathlib import Path
def create_enhanced_negative_samples(output_dir, sample_rate=16000, duration=1.2, num_samples=200):
    """
    创建增强的负例样本，专门解决静音误触发问题，可配置样本数量
    
    Args:
        output_dir: 输出目录
        sample_rate: 采样率
        duration: 音频时长（秒）
        num_samples: 每种类型的样本数量，第4和5种为num_samples//2
    """
    negative_dir = Path(output_dir) / "negative"
    negative_dir.mkdir(parents=True, exist_ok=True)
    
    samples = int(duration * sample_rate)
    half_samples = num_samples // 2
    
    print(f"🔧 生成针对静音误触发的负例样本 (每种类型 {num_samples} 个)...")
    
    # 1. 绝对静音样本 (最重要！)
    print(f"生成 {num_samples} 个绝对静音样本...")
    for i in range(num_samples):
        silence = np.zeros(samples, dtype=np.float32)
        sf.write(negative_dir / f"silence_{i:03d}.wav", silence, sample_rate)
    
    # 2. 极小噪音样本 (模拟TTS的背景噪音)
    print(f"生成 {num_samples} 个极小噪音样本...")
    for i in range(num_samples):
        tiny_noise = np.random.normal(0, 0.001, samples).astype(np.float32)
        sf.write(negative_dir / f"tiny_noise_{i:03d}.wav", tiny_noise, sample_rate)
    
    # 3. 小噪音样本
    print(f"生成 {num_samples} 个小噪音样本...")
    for i in range(num_samples):
        small_noise = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(negative_dir / f"small_noise_{i:03d}.wav", small_noise, sample_rate)
    
    # 4. 模拟TTS静音模式（前后静音，中间极小信号）
    print(f"生成 {half_samples} 个TTS静音模式样本...")
    for i in range(half_samples):
        tts_like = np.zeros(samples)
        middle_start = samples // 4
        middle_end = 3 * samples // 4
        # 中间部分加入极小的随机信号，模拟TTS的微弱背景
        tts_like[middle_start:middle_end] = np.random.normal(0, 0.002, middle_end - middle_start)
        sf.write(negative_dir / f"tts_like_negative_{i:03d}.wav", tts_like.astype(np.float32), sample_rate)
    
    # 5. 渐变噪音（从静音到小噪音的过渡）
    print(f"生成 {half_samples} 个渐变噪音样本...")
    for i in range(half_samples):
        gradient_noise = np.zeros(samples)
        for j in range(samples):
            # 噪音强度从0渐变到0.005
            noise_level = (j / samples) * 0.005
            gradient_noise[j] = np.random.normal(0, noise_level)
        sf.write(negative_dir / f"gradient_noise_{i:03d}.wav", gradient_noise.astype(np.float32), sample_rate)
    
    total_samples = num_samples * 3 + half_samples * 2
    print(f"✅ 总共生成了 {total_samples} 个针对性负例样本到 {negative_dir}")
    print("📊 分布：")
    print(f"  - {num_samples} 个绝对静音样本")
    print(f"  - {num_samples} 个极小噪音样本")
    print(f"  - {num_samples} 个小噪音样本")
    print(f"  - {half_samples} 个TTS静音模式样本")
    print(f"  - {half_samples} 个渐变噪音样本")
