"""
Model Training Module (DEPRECATED)

This module has been merged into train.py for a unified training pipeline.
Please use train.py instead.

Usage:
    python train.py
"""

import sys
import os

# Redirect to train.py
if __name__ == '__main__':
    print("=" * 60)
    print("NOTICE: model_training.py has been merged into train.py")
    print("=" * 60)
    print("\nPlease run: python train.py")
    print("\nRedirecting to train.py...")
    print("=" * 60)
    
    # Import and run train
    sys.path.insert(0, os.path.dirname(__file__))
    import train
    train.train_and_compare_models()
