#!/usr/bin/env python3
"""
Test script for the wake word training pipeline
"""

import os
import tempfile
import shutil
from pathlib import Path
import argparse

# Import the pipeline function
from pipeline import run_pipeline


#  python pipeline.py --data_dir ./my_dataset --wake_word "你好小智" --epochs 10 --batch_size 32  resplit_data

def test_pipeline():
    """Test the complete pipeline with sample data"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing pipeline in temporary directory: {temp_dir}")
        
        # Create test arguments
        args = argparse.Namespace(
            data_dir=os.path.join(temp_dir, "dataset"),
            wake_word="xiaozhi",
            output_dir=os.path.join(temp_dir, "output"),
            deploy_dir=os.path.join(temp_dir, "deploy"),
            create_new_data=True,  # Generate new sample data
            resplit_data=True,     # Split the data
            
            # Model parameters (use small values for quick testing)
            sample_rate=16000,
            n_mfcc=13,
            window_size_ms=30,
            stride_ms=30,
            n_frames=40,
            model_dim=16,  # Smaller model for testing
            threshold=0.5,
            averaging_frames=5,
            
            # Training parameters (small values for quick testing)  
            batch_size=8,
            epochs=2,  # Very few epochs for testing
            learning_rate=0.001,
            num_workers=1,  # Single worker for testing
            no_cuda=True  # Force CPU for testing
        )
        
        try:
            # Run the pipeline
            output_dir = run_pipeline(args)
            print(f"Pipeline completed successfully!")
            print(f"Output directory: {output_dir}")
            
            # Check if key files were created
            expected_files = [
                "config.json",
                "best_model.pth",
                "wakenet_model.onnx",
                "test_metrics.json",
                "predictions.png",
                "pipeline.log"
            ]
            
            for file_name in expected_files:
                file_path = os.path.join(output_dir, file_name)
                if os.path.exists(file_path):
                    print(f"✓ {file_name} created successfully")
                else:
                    print(f"✗ {file_name} missing")
            
            # Check deployment directory
            deploy_path = os.path.join(args.deploy_dir, args.wake_word)
            if os.path.exists(deploy_path):
                print(f"✓ Deployment files created at {deploy_path}")
            else:
                print(f"✗ Deployment directory missing")
                
            return True
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Tests failed!")
        exit(1) 