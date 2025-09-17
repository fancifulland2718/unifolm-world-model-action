# UnifoLM-WMA Codebase Architecture and Usage Flow Analysis

## Project Overview

UnifoLM-WMA-0 is a World-Model-Action (WMA) framework for robotic learning under the UnifoLM family. It's designed specifically for general-purpose robot learning across multiple robotic embodiment types.

**Core Features:**
- **World Model**: Understands physical interactions between robots and environments
- **Dual Functions**:
  - **Simulation Engine**: Acts as interactive simulator generating synthetic data for robot learning
  - **Policy Enhancement**: Connects with action head to optimize decision-making by predicting future interactions

## Architecture Overview

### Directory Structure
```
unifolm-world-model-action/
├── assets/                      # Media assets (GIFs, images, demo videos)
├── configs/                     # Configuration files
│   ├── inference/              # Inference configurations
│   │   ├── base_model_inference.yaml
│   │   └── world_model_interaction.yaml
│   └── train/                  # Training configurations
│       ├── config.yaml         # Main training config
│       └── meta.json          # Data metadata
├── examples/                   # Example data and inputs
├── external/                   # External package dependencies
├── prepare_data/               # Data preprocessing scripts
│   └── prepare_training_data.py
├── scripts/                    # Main execution scripts
│   ├── trainer.py             # Training script
│   └── evaluation/            # Evaluation scripts
│       ├── base_model_inference.py
│       ├── world_model_interaction.py
│       └── eval_utils.py
└── src/unifolm_wma/           # Core Python package
    ├── data/                  # Data loading and processing
    │   ├── base.py
    │   ├── wma_data.py        # Main dataset class
    │   ├── normolize.py       # Data normalization
    │   └── utils.py
    ├── models/                # Model architectures
    │   ├── ddpms.py           # Main diffusion model class
    │   ├── autoencoder.py     # Autoencoder
    │   ├── diffusion_head/    # Diffusion head components
    │   └── samplers/          # Samplers
    ├── modules/               # Custom modules
    │   ├── attention.py       # Attention mechanisms
    │   ├── encoders/          # Encoders
    │   ├── networks/          # Network components
    │   └── vision/            # Vision components
    └── utils/                 # Utility functions
```

### Core Components

#### 1. Data Layer (`src/unifolm_wma/data/`)
- **WMAData Class** (`wma_data.py`): Main dataset class handling multimodal data
  - Supports video, robot state, and action data
  - Data format: H5 files for states/actions, MP4 for videos
  - Normalization and unnormalization processing

#### 2. Model Layer (`src/unifolm_wma/models/`)
- **DDPM** (`ddpms.py`): Denoising Diffusion Probabilistic Model inheriting from PyTorch Lightning
  - Supports two modes: decision-making mode and simulation mode
  - Integrates vision encoder and action head
  - Uses DDIM sampler for inference

- **Diffusion Head** (`diffusion_head/`): Specialized network for action prediction
  - Conditional UNet1D architecture
  - Supports positional encoding
  - EMA model for stable training

#### 3. Module Layer (`src/unifolm_wma/modules/`)
- **Attention Mechanisms** (`attention.py`): Multiple attention implementations
  - CrossAttention, SpatialTransformer, TemporalTransformer
  - xformers optimization support
  - Relative positional encoding

- **Vision Components** (`vision/`): Vision encoders
  - DINO-SigLIP ViT architecture
  - Base vision modules

### Configuration System

#### Training Configuration (`configs/train/config.yaml`)
```yaml
model:
  target: unifolm_wma.models.ddpms.LatentVisualDiffusion
  params:
    # Diffusion model parameters
    timesteps: 1000
    linear_start: 0.00085
    linear_end: 0.012
    
    # Input dimension configuration
    agent_state_dim: 16      # Robot state dimension (max 16 DoF)
    agent_action_dim: 16     # Robot action dimension
    
    # Mode selection
    decision_making_only: true  # Decision-only mode vs joint training
    
    # Observation steps
    n_obs_steps_imagen: 2    # Image observation steps
    n_obs_steps_acting: 2    # Action observation steps
```

#### Data Metadata (`configs/train/meta.json`)
```json
{
    "obs": {
        "image": {"shape": [3, 320, 512], "type": "rgb"},
        "agent_pos": {"shape": [16], "type": "low_dim"}
    },
    "action": {"shape": [16]}
}
```

## Typical Usage Flow

### 1. Training Workflow

#### A. Data Preparation
```bash
# Preprocess training data
python prepare_data/prepare_training_data.py \
    --source_dir /path/to/raw/data \
    --target_dir /path/to/processed/data \
    --dataset_name your_dataset
```

**Required Data Structure:**
```
dataset_dir/
├── videos/
│   └── dataset_name/
│       └── camera_view/
│           ├── 0.mp4
│           └── 1.mp4
├── transitions/
│   └── dataset_name/
│       ├── 0.h5          # Contains state and action data
│       └── 1.h5
└── dataset_name.csv      # Metadata index
```

#### B. Configuration Setup
1. Update model parameters in `configs/train/config.yaml`
2. Set pretrained checkpoint path
3. Configure data directories and dataset weights

#### C. Training Execution
```bash
# Single GPU training
python scripts/trainer.py \
    --base configs/train/config.yaml \
    --name experiment_name \
    --train

# Multi-GPU distributed training
torchrun --nproc_per_node=8 scripts/trainer.py \
    --base configs/train/config.yaml \
    --name experiment_name \
    --train
```

#### Training Strategy (3 Steps):
1. **Step 1**: Fine-tune video generation model as world model on Open-X dataset
2. **Step 2**: Post-train in decision-making mode on downstream task dataset
3. **Step 3**: Post-train in simulation mode on downstream task dataset

### 2. Inference Workflow

#### A. Base Model Inference
```bash
python scripts/evaluation/base_model_inference.py \
    --config configs/inference/base_model_inference.yaml \
    --ckpt_path /path/to/checkpoint \
    --prompt_dir /path/to/prompts \
    --savedir /path/to/output
```

#### B. World Model Interactive Inference
```bash
python scripts/evaluation/world_model_interaction.py \
    --config configs/inference/world_model_interaction.yaml \
    --ckpt_path /path/to/checkpoint \
    --prompt_dir /path/to/prompts \
    --dataset dataset_name \
    --savedir /path/to/output
```

### 3. Core Workflow

#### Training Phase Flow:
1. **Data Loading**: WMAData class loads multimodal data (video, state, action)
2. **Model Initialization**: Instantiate LatentVisualDiffusion model
3. **Forward Pass**: 
   - Encode input images to latent space
   - Process language instructions and robot states
   - Diffusion process predicts future frames and actions
4. **Loss Computation**: Calculate diffusion loss and action prediction loss
5. **Backward Pass**: Update parameters using AdamW optimizer

#### Inference Phase Flow:
1. **Model Loading**: Load pretrained checkpoint
2. **Input Processing**: Process observation images, language instructions, robot states
3. **DDIM Sampling**: 
   - Use DDIM sampler to generate future frames
   - Simultaneously predict robot action sequences
4. **Post-processing**: Decode latent representations to pixel space
5. **Output**: Generated video frames and action sequences

### 4. Key Technical Features

#### Multimodal Fusion:
- **Vision**: RGB image sequences with multi-view support
- **Language**: Natural language instruction encoding
- **State**: Robot joint positions and velocities
- **Action**: Robot control commands

#### Dual-Mode Architecture:
- **Decision Mode**: Directly predicts optimal actions
- **Simulation Mode**: Predicts state transitions and future observations

#### Diffusion Architecture:
- DDPM-based generative model
- Supports conditional and unconditional generation
- Joint modeling of vision-language-action

### 5. Model Checkpoints

| Model | Description | Purpose |
|-------|-------------|---------|
| UnifoLM-WMA-0_Base | Fine-tuned on Open-X dataset | Base world model |
| UnifoLM-WMA-0_Dual | Dual-mode trained on Unitree datasets | Production-ready model |

### 6. Dependencies

**Core Dependencies:**
- Python 3.10.18
- PyTorch 2.3.1 + torchvision 0.18.1
- PyTorch Lightning 1.9.3
- xformers 0.0.27 (attention optimization)
- transformers 4.40.1
- Others: einops, omegaconf, pandas, etc.

## Key Implementation Details

### Data Flow Architecture
```
Input: RGB Images + Language Instructions + Robot States
    ↓
Vision Encoder (DINO-SigLIP ViT)
    ↓
Latent Space Encoding
    ↓
World Model (Conditional Diffusion)
    ├── Decision Mode → Action Head → Robot Actions
    └── Simulation Mode → State Head → Future States
    ↓
Future Frame Generation
```

### Training Modes
1. **Decision-Making Mode**: 
   - Input: Current observation + language instruction
   - Output: Next action sequence
   - Loss: Action prediction + diffusion loss

2. **Simulation Mode**:
   - Input: Current state + action sequence
   - Output: Future state sequence + future frames
   - Loss: State prediction + frame reconstruction loss

### Inference Pipeline
1. **Conditioning**: Prepare multimodal conditions
2. **Noise Initialization**: Random noise in latent space
3. **Iterative Denoising**: DDIM sampling with guidance
4. **Decoding**: Convert latents to pixel space
5. **Action Extraction**: Extract action sequences

## Summary

UnifoLM-WMA is a comprehensive end-to-end robotic learning framework that integrates state-of-the-art diffusion model technology, multimodal learning, and world model approaches. Its modular design supports flexible configuration for various robotic application scenarios. Through the dual functions of the world model (simulation and policy enhancement), the framework effectively utilizes data and improves robot decision-making performance.

The architecture demonstrates strong potential for scaling to different robotic embodiments and tasks, making it a valuable contribution to the field of general-purpose robotics.