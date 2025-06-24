import os
import json
import torch
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic
import shutil
from pathlib import Path

def convert_to_onnx(model, input_shape, output_path):
    """将PyTorch模型转换为ONNX格式"""
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 验证ONNX模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # 比较输出
    ort_session = onnxruntime.InferenceSession(output_path)
    
    # 获取输入和输出名
    inputs = ort_session.get_inputs()[0].name
    outputs = ort_session.get_outputs()[0].name
    
    # 比较输出
    ort_inputs = {inputs: dummy_input.numpy()}
    ort_outputs = ort_session.run([outputs], ort_inputs)[0]
    
    pytorch_outputs = model(dummy_input).detach().numpy()
    
    # 验证输出相似
    np.testing.assert_allclose(pytorch_outputs, ort_outputs, rtol=1e-3, atol=1e-5)
    
    print(f"ONNX模型已导出到 {output_path} 并验证成功")
    return output_path

def quantize_model(onnx_path, output_dir, config):
    """量化ONNX模型并转换为ESP-DL格式"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 量化ONNX模型
    quantized_path = os.path.join(output_dir, "wakenet_quantized.onnx")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=np.int8
    )
    
    # 复制ONNX模型
    shutil.copy(quantized_path, os.path.join(output_dir, "model.onnx"))
    
    # 生成ESP-DL配置文件
    esp_config = {
        "wake_word": config["wake_word"],
        "sample_rate": config["sample_rate"],
        "window_size_ms": config["window_size_ms"],
        "stride_ms": config["stride_ms"],
        "n_mfcc": config["n_mfcc"],
        "n_frames": config["n_frames"],
        "threshold": config["threshold"],
        "averaging_frames": config["averaging_frames"],
        "model_version": "wakenet_v1",
        "esp_platform": "esp32s3",
        "input_node_name": "input",
        "output_node_name": "output"
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(esp_config, f, indent=2)
    
    # 创建README文件
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"# WakeNet 唤醒词模型 - {config['wake_word']}\n\n")
        f.write("此模型为ESP32-S3优化的轻量级唤醒词模型。\n\n")
        f.write("## 使用方法\n\n")
        f.write("1. 将整个文件夹复制到ESP-IDF项目的模型目录\n")
        f.write("2. 使用ESP-DL的API加载模型\n")
    
    print(f"量化模型并转换为ESP-DL格式，保存在 {output_dir}")
