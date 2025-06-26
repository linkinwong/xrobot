import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import shutil
import json
import logging

from model import DilatedWakeNet
from dataset import WakeWordDataset, create_sample_dataset
from utils import setup_logger, save_config, load_config, calculate_model_size, visualize_predictions
from convert import convert_to_onnx, quantize_model
from create_enhanced_negative_samples import create_enhanced_negative_samples

# python pipeline.py --data_dir ./my_dataset --wake_word "nihaoxiaoqi" --epochs 2 --batch_size 8 --create_new_data --resplit_data
# python pipeline.py --data_dir ./my_dataset --wake_word "xiaoqi" --epochs 2 --batch_size 64  --resplit_data --num_workers 20
# nohup  python pipeline.py --data_dir ./my_dataset --wake_word "xiaoqi" --epochs 2 --batch_size 64  --resplit_data --num_workers 20  > train_log_june_24_21_31 2>&1 &

all_wake_words = ["你好小灵", "你好小七", "嗨, 小七", "嗨,小灵", "嘿,小七",  "嘿,小灵", "小七小七", "小灵小灵"] 

def generate_and_split_dataset(data_root, wake_word, logger, create_new_data=False, resplit_data=False, sample_rate=16000):
    """
    Generate sample dataset and split into train/val/test sets
    
    Args:
        data_root: Root directory for dataset
        wake_word: Target wake word
        logger: Logger instance
        create_new_data: Whether to generate new sample data
        resplit_data: Whether to re-split existing data
        sample_rate: Audio sample rate
    """
    data_path = Path(data_root)
    
    # Create directories
    raw_data_dir = data_path / "raw"
    
    # Step 1: Generate sample data if needed
    if create_new_data or not raw_data_dir.exists():
        logger.info("Generating sample dataset...")
        if raw_data_dir.exists():
            shutil.rmtree(raw_data_dir)
        create_sample_dataset(str(raw_data_dir), sample_rate=sample_rate)
        logger.info(f"Sample dataset created at {raw_data_dir}")
    
    # Step 2: Check if split already exists and is valid
    splits_exist = all((data_path / split).exists() for split in ["train", "val", "test"])
    
    if not resplit_data and splits_exist:
        logger.info(f"Dataset splits already exist at {data_root}")
        return
    
    # Step 3: Split data into train/val/test (7:2:1)
    logger.info("Splitting dataset into train/val/test (7:2:1)...")
    
    # Clean existing splits if re-splitting
    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        
        # Create new structure
        os.makedirs(split_dir / "positive", exist_ok=True)
        os.makedirs(split_dir / "negative", exist_ok=True)
    
    # Get all files
    pos_files = sorted(list((raw_data_dir / "positive").glob("*.wav")))
    neg_files = sorted(list((raw_data_dir / "negative").glob("*.wav")))
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Shuffle files
    pos_indices = np.random.permutation(len(pos_files))
    neg_indices = np.random.permutation(len(neg_files))
    
    pos_files = [pos_files[i] for i in pos_indices]
    neg_files = [neg_files[i] for i in neg_indices]
    
    # Calculate split sizes
    pos_train_size = int(len(pos_files) * 0.7)
    pos_val_size = int(len(pos_files) * 0.2)
    
    neg_train_size = int(len(neg_files) * 0.7)
    neg_val_size = int(len(neg_files) * 0.2)
    
    # Split positive files
    pos_train = pos_files[:pos_train_size]
    pos_val = pos_files[pos_train_size:pos_train_size + pos_val_size]
    pos_test = pos_files[pos_train_size + pos_val_size:]
    
    # Split negative files
    neg_train = neg_files[:neg_train_size]
    neg_val = neg_files[neg_train_size:neg_train_size + neg_val_size]
    neg_test = neg_files[neg_train_size + neg_val_size:]
    
    # Copy files to respective directories
    def copy_files(file_list, dest_dir, category):
        for file_path in file_list:
            dest_path = dest_dir / category / file_path.name
            shutil.copy2(file_path, dest_path)
    
    # Copy train files
    copy_files(pos_train, data_path / "train", "positive")
    copy_files(neg_train, data_path / "train", "negative")
    
    # Copy validation files
    copy_files(pos_val, data_path / "val", "positive")
    copy_files(neg_val, data_path / "val", "negative")
    
    # Copy test files
    copy_files(pos_test, data_path / "test", "positive")
    copy_files(neg_test, data_path / "test", "negative")
    
    logger.info(f"Dataset split completed:")
    logger.info(f"  Train: {len(pos_train)} positive, {len(neg_train)} negative")
    logger.info(f"  Val: {len(pos_val)} positive, {len(neg_val)} negative")
    logger.info(f"  Test: {len(pos_test)} positive, {len(neg_test)} negative")


def enhance_negative_samples_for_silence_fix(data_root):
    """
    专门为解决静音误触发问题增强负例样本
    """
    create_enhanced_negative_samples(
        output_dir=f"{data_root}/raw",  # 添加到现有数据中
        sample_rate=16000,
        duration=1.2  # 与训练配置一致
    )
    print("🎯 静音误触发修复：负例增强完成！")

def train_model_directly(model, train_loader, val_loader, device, epochs, learning_rate, output_dir, logger):
    """
    Train the model directly without using the train.py function
    """
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for mfccs, labels in pbar:
            mfccs = mfccs.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            logits = model(mfccs).squeeze(-1)  # Remove last dimension
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for mfccs, labels in pbar:
                mfccs = mfccs.to(device)
                labels = labels.to(device).float()
                
                logits = model(mfccs).squeeze(-1)  # Remove last dimension
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate adjustment
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(output_dir, "best_model.pth"))
            logger.info(f"Saved best model, validation loss: {val_loss:.4f}")
    
    logger.info("Training completed!")
    
    # Convert to ONNX
    logger.info("Converting to ONNX model...")
    onnx_path = os.path.join(output_dir, "wakenet_model.onnx")
    
    # Create a sample input for ONNX conversion and move to same device as model
    sample_input = torch.randn(1, 40, 13).to(device)  # (batch, frames, features)
    model.eval()
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['mfcc'],
            output_names=['logits'],
            dynamic_axes={
                'mfcc': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
    
    logger.info(f"ONNX model saved to {onnx_path}")

def test_model(model, test_loader, device, logger):
    """
    Test the model on the test dataset
    """
    logger.info("Testing model on test dataset...")
    
    model.eval()
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs = mfccs.to(device)
            labels = labels.to(device).float()
            
            logits = model(mfccs).squeeze(-1)  # Remove last dimension
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # Update metrics
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Calculate confusion matrix metrics
            true_positives += ((preds == 1) & (labels == 1)).sum().item()
            false_positives += ((preds == 1) & (labels == 0)).sum().item()
            true_negatives += ((preds == 0) & (labels == 0)).sum().item()
            false_negatives += ((preds == 0) & (labels == 1)).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    logger.info(f"Confusion Matrix:")
    logger.info(f"  True Positives: {true_positives}")
    logger.info(f"  False Positives: {false_positives}")
    logger.info(f"  True Negatives: {true_negatives}")
    logger.info(f"  False Negatives: {false_negatives}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives
    }

def run_pipeline(args):
    """
    Run the complete wake word model pipeline
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.wake_word}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger("pipeline")
    log_file_handler = logging.FileHandler(os.path.join(output_dir, "pipeline.log"))
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(log_file_handler)
    
    logger.info(f"Starting wake word model pipeline for '{args.wake_word}'")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Create config first
    config = {
        "sample_rate": args.sample_rate,
        "n_mfcc": args.n_mfcc,
        "window_size_ms": args.window_size_ms,
        "stride_ms": args.stride_ms,
        "n_frames": args.n_frames,
        "model_dim": args.model_dim,
        "wake_word": args.wake_word,
        "threshold": args.threshold,
        "averaging_frames": args.averaging_frames
    }
    
    config_path = os.path.join(output_dir, "config.json")
    save_config(config, config_path)
    logger.info(f"Saved config to {config_path}")
    
    # Step 2: Generate and split dataset
    generate_and_split_dataset(
        args.data_dir, 
        args.wake_word, 
        logger, 
        create_new_data=args.create_new_data,
        resplit_data=args.resplit_data,
        sample_rate=config["sample_rate"]
    )
    
    # Step 3: Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Step 4: Load datasets
    logger.info("Loading datasets...")
    
    # Training dataset
    train_dataset = WakeWordDataset(
        os.path.join(args.data_dir, "train"), 
        wake_word=args.wake_word,
        mode="train", 
        n_mfcc=config["n_mfcc"],
        sample_rate=config["sample_rate"],
        window_size_ms=config["window_size_ms"],
        stride_ms=config["stride_ms"],
        n_frames=config["n_frames"],
        train_ratio=0.99999  # Use virtually all data since we pre-split
    )
    
    # Validation dataset - use mode="train" to get all data from val folder
    val_dataset = WakeWordDataset(
        os.path.join(args.data_dir, "val"),
        wake_word=args.wake_word,
        mode="train",  # Use train mode to avoid further splitting
        n_mfcc=config["n_mfcc"],
        sample_rate=config["sample_rate"],
        window_size_ms=config["window_size_ms"],
        stride_ms=config["stride_ms"],
        n_frames=config["n_frames"],
        train_ratio=0.99999  # Use virtually all data since we pre-split
    )
    
    # Test dataset - use mode="train" to get all data from test folder
    test_dataset = WakeWordDataset(
        os.path.join(args.data_dir, "test"),
        wake_word=args.wake_word,
        mode="train",  # Use train mode to avoid further splitting
        n_mfcc=config["n_mfcc"],
        sample_rate=config["sample_rate"],
        window_size_ms=config["window_size_ms"],
        stride_ms=config["stride_ms"],
        n_frames=config["n_frames"],
        train_ratio=0.99999  # Use virtually all data since we pre-split
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Step 5: Create model
    model = DilatedWakeNet(
        input_dim=config["n_mfcc"], 
        model_dim=config["model_dim"]
    ).to(device)
    
    # Calculate and log model size
    model_size = calculate_model_size(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model size: {model_size:.2f} MB")
    logger.info(f"Total parameters: {total_params:,}")
    
    # Step 6: Train model
    logger.info("Starting model training...")
    
    # Train the model directly instead of calling train function
    train_model_directly(
        model, train_loader, val_loader, device, 
        args.epochs, args.learning_rate, output_dir, logger
    )
    
    # Step 7: Load best model for evaluation
    best_model_path = os.path.join(output_dir, "best_model.pth")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from {best_model_path}")
    logger.info(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Step 8: Test the model
    test_metrics = test_model(model, test_loader, device, logger)
    
    # Save test metrics
    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Step 9: Visualize predictions
    logger.info("Generating prediction visualizations...")
    fig = visualize_predictions(model, test_dataset, num_samples=min(5, len(test_dataset)))
    fig.savefig(os.path.join(output_dir, "predictions.png"))
    logger.info(f"Saved predictions visualization to {output_dir}/predictions.png")
    
    # Step 10: Copy final files to output path for deployment
    deploy_dir = os.path.join(args.deploy_dir, args.wake_word)
    if not os.path.exists(deploy_dir):
        os.makedirs(deploy_dir)
    
    # Copy the ONNX model and config to the deployment directory
    shutil.copy(
        os.path.join(output_dir, "wakenet_model.onnx"),
        os.path.join(deploy_dir, "wakenet_model.onnx")
    )
    shutil.copy(
        os.path.join(output_dir, "config.json"),
        os.path.join(deploy_dir, "config.json")
    )
    
    # Copy ESP-DL model if it exists
    esp_model_dir = os.path.join(output_dir, "esp_model")
    if os.path.exists(esp_model_dir):
        esp_deploy_dir = os.path.join(deploy_dir, "esp_model")
        if os.path.exists(esp_deploy_dir):
            shutil.rmtree(esp_deploy_dir)
        shutil.copytree(esp_model_dir, esp_deploy_dir)
    
    logger.info(f"Copied deployment files to {deploy_dir}")
    logger.info("Pipeline completed successfully!")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete pipeline for wake word model training")
    
    # Dataset parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory containing audio files")
    parser.add_argument("--wake_word", type=str, required=True, help="Target wake word to detect")
    parser.add_argument("--create_new_data", action="store_true", help="Generate new sample dataset (default: False)")
    parser.add_argument("--resplit_data", action="store_true", help="Re-split existing dataset (default: False)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save all outputs")
    parser.add_argument("--deploy_dir", type=str, default="./models", help="Directory to save deployment files")
    
    # Model parameters
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCC features")
    parser.add_argument("--window_size_ms", type=int, default=30, help="Window size in milliseconds")
    parser.add_argument("--stride_ms", type=int, default=30, help="Stride in milliseconds")
    parser.add_argument("--n_frames", type=int, default=40, help="Number of frames per sample")
    parser.add_argument("--model_dim", type=int, default=32, help="Model dimension")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--averaging_frames", type=int, default=5, help="Number of frames for averaging")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.deploy_dir, exist_ok=True)
    
    # Run the pipeline
    output_dir = run_pipeline(args)