# DMSF-Net: Synergistic Integration of Deformable Convolution and Attention Gating for Medical Image Segmentation

## Abstract
This repository contains the official PyTorch implementation of **DMSF-Net**. 
Medical image segmentation faces persistent challenges in anatomical heterogeneity, low-contrast boundaries, and multi-scale pathology presentations. We propose DMSF-Net, an integrated architecture featuring Hybrid Convolution-Attention (HCA), Multi-scale Dynamic Fusion (MDF), and Gated Hierarchical Skip Connections (HSC). 

## Project Structure
The code is organized into modular components to ensure extensibility and maintainability:

- `modules.py`: Contains the core architectural innovations (HCA, MDF, HSC).
- `network.py`: Assembles the ResNet-50 encoder with the proposed modules.
- `preprocess.py`: Handles data preparation, mask generation from polygons, and ROI cropping.
- `train.py`: The main training pipeline with metric logging.
- `config.py`: Centralized configuration for hyperparameters and paths.

## Usage

### 1. Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt