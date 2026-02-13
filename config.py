# coding=utf8
import torch


class Config:
    # System settings
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data paths (Placeholder paths requiring user configuration)
    TRAIN_IMG_DIR = "./data/train/images"
    TRAIN_MASK_DIR = "./data/train/masks"
    VAL_IMG_DIR = "./data/valid/images"
    VAL_MASK_DIR = "./data/valid/masks"
    TEST_IMG_DIR = "./data/test/images"
    TEST_MASK_DIR = "./data/test/masks"

    # Hyperparameters
    IMG_SIZE = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    NUM_CLASSES = 1

    # Model saving
    SAVE_PATH = "./checkpoints/dmsf_net_best.pth"