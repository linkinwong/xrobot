# 实时唤醒词检测推理说明

## 功能简介

`infer_rt.py` 是一个实时唤醒词检测推理脚本，可以使用训练好的 ONNX 模型进行实时音频流唤醒词检测。

## 主要特性

- 🎯 **实时检测**: 支持实时音频流输入和唤醒词检测
- 🔧 **跨平台**: 使用 PyAudio，支持 Linux 和 macOS
- 📊 **可调参数**: 推理间隔、平滑帧数等重要参数均可调节
- 📈 **性能监控**: 实时显示推理时间和检测统计信息
- 🎚️ **交互控制**: 支持暂停/恢复检测，实时查看统计信息

## 安装依赖

```bash
pip install -r requirements_infer.txt
```

### macOS 额外配置

在 macOS 上，如果遇到 PyAudio 安装问题：

```bash
# 使用 Homebrew 安装 portaudio
brew install portaudio

# 然后安装 PyAudio
pip install pyaudio
```

### Linux 额外配置

在 Ubuntu/Debian 上：

```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

## 使用方法

### 基本用法

```bash
# 使用默认参数运行
python infer_rt.py --model_dir ./output/xiaoqi_20250625_105407

# 查看可用音频设备
python infer_rt.py --list_devices

# 指定音频设备
python infer_rt.py --model_dir ./output/xiaoqi_20250625_105407 --device 1
```

### 参数调节

```bash
# 调整推理间隔（毫秒）- 越小检测越频繁但CPU占用越高
python infer_rt.py --model_dir ./output/xiaoqi_20250625_105407 --inference_interval 50

# 调整平滑帧数 - 越大检测越稳定但响应越慢
python infer_rt.py --model_dir ./output/xiaoqi_20250625_105407 --averaging_frames 10

# 详细输出模式 - 显示所有推理结果
python infer_rt.py --model_dir ./output/xiaoqi_20250625_105407 --verbose
```

### 组合参数示例

```bash
# 高频检测 + 详细输出
python infer_rt.py \
    --model_dir ./output/xiaoqi_20250625_105407 \
    --device 1 \
    --inference_interval 50 \
    --averaging_frames 3 \
    --verbose
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_dir` | str | `./output/xiaoqi_20250625_105407` | 模型目录路径 |
| `--device` | int | None | 音频设备索引（None=默认设备） |
| `--list_devices` | flag | False | 列出可用音频设备 |
| `--inference_interval` | int | 100 | 推理间隔（毫秒） |
| `--averaging_frames` | int | 5 | 平滑预测的帧数 |
| `--verbose` | flag | False | 详细输出模式 |

## 交互命令

运行时可以使用以下键盘命令：

- `q` + Enter: 退出程序
- `p` + Enter: 暂停/恢复检测
- `s` + Enter: 显示统计信息

## 输出说明

### 正常检测输出
```
🎯 [14:30:25] 检测到唤醒词! 置信度: 0.856, 平滑: 0.743
```

### 详细模式输出
```
   [14:30:20] 置信度: 0.123, 平滑: 0.145
   [14:30:21] 置信度: 0.089, 平滑: 0.134
🎯 [14:30:25] 检测到唤醒词! 置信度: 0.856, 平滑: 0.743
```

### 统计信息输出
```
统计信息:
  总推理次数: 1250
  平均推理时间: 15.32ms
  最大推理时间: 23.45ms
  最小推理时间: 12.10ms
```

## 性能调优

### 推理间隔优化

- **高响应性**: `--inference_interval 50` (每50ms检测一次)
- **平衡模式**: `--inference_interval 100` (默认，每100ms检测一次)
- **低功耗**: `--inference_interval 200` (每200ms检测一次)

### 平滑帧数优化

- **快速响应**: `--averaging_frames 3` (检测延迟约300ms)
- **平衡模式**: `--averaging_frames 5` (默认，检测延迟约500ms)  
- **稳定检测**: `--averaging_frames 10` (检测延迟约1000ms)

## 故障排除

### 常见问题

1. **音频设备错误**
   ```bash
   # 先查看可用设备
   python infer_rt.py --list_devices
   # 然后指定正确的设备索引
   python infer_rt.py --device X
   ```

2. **权限问题（macOS）**
   - 在系统偏好设置中授予麦克风访问权限

3. **依赖包问题**
   ```bash
   # 重新安装音频相关包
   pip uninstall pyaudio
   pip install pyaudio
   ```

4. **ONNX 运行时错误**
   ```bash
   # 检查 ONNX 版本兼容性
   pip install onnxruntime --upgrade
   ```

### 性能问题

1. **推理时间过长**
   - 增大 `--inference_interval` 降低检测频率
   - 检查CPU负载和内存使用情况

2. **检测不敏感**
   - 减小 `--averaging_frames` 提高响应速度
   - 检查音频输入音量和质量

3. **误检测过多**
   - 增大 `--averaging_frames` 提高稳定性
   - 考虑重新训练模型或调整阈值

## 技术架构

```
音频输入 → 环形缓冲区 → MFCC特征提取 → ONNX模型推理 → 置信度平滑 → 检测结果
    ↓          ↓             ↓             ↓            ↓          ↓
  PyAudio   AudioBuffer   MFCCExtractor  ONNXRuntime   历史平滑    阈值判断
```

## 注意事项

1. **模型路径**: 确保模型目录包含 `wakenet_model.onnx` 和 `config.json` 文件
2. **音频质量**: 建议使用高质量麦克风，避免环境噪音干扰
3. **系统资源**: 实时推理会持续占用CPU和内存资源
4. **网络依赖**: 推理完全在本地运行，无需网络连接

## 扩展功能

可以基于此脚本进行以下扩展：

1. **多唤醒词检测**: 加载多个模型同时检测
2. **音频录制**: 检测到唤醒词时自动录制后续音频
3. **回调机制**: 检测成功时触发自定义动作
4. **GUI界面**: 创建图形化界面显示检测状态
5. **日志记录**: 记录检测历史和统计信息到文件 