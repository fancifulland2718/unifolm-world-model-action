# UnifoLM-WMA 代码库架构与使用流程分析

## 项目概述

UnifoLM-WMA-0 是一个基于世界模型-动作(World-Model-Action, WMA)的机器人学习框架，属于 UnifoLM 系列。该项目专为通用机器人学习设计，支持多种机器人实体类型。

核心特点：
- **世界模型**：理解机器人与环境的物理交互
- **双重功能**：
  - **仿真引擎**：作为交互式模拟器生成机器人学习的合成数据
  - **策略增强**：连接动作头，通过预测未来交互过程优化决策性能

## 代码架构

### 目录结构
```
unifolm-world-model-action/
├── assets/                      # 媒体资源 (GIF、图片、演示视频)
├── configs/                     # 配置文件
│   ├── inference/              # 推理配置
│   │   ├── base_model_inference.yaml
│   │   └── world_model_interaction.yaml
│   └── train/                  # 训练配置
│       ├── config.yaml         # 主训练配置
│       └── meta.json          # 数据元信息
├── examples/                   # 示例数据和输入
├── external/                   # 外部包依赖
├── prepare_data/               # 数据预处理脚本
│   └── prepare_training_data.py
├── scripts/                    # 主要执行脚本
│   ├── trainer.py             # 训练脚本
│   └── evaluation/            # 评估脚本
│       ├── base_model_inference.py
│       ├── world_model_interaction.py
│       └── eval_utils.py
└── src/unifolm_wma/           # 核心Python包
    ├── data/                  # 数据加载和处理
    │   ├── base.py
    │   ├── wma_data.py        # 主要数据集类
    │   ├── normolize.py       # 数据归一化
    │   └── utils.py
    ├── models/                # 模型架构
    │   ├── ddpms.py           # 扩散模型主类
    │   ├── autoencoder.py     # 自编码器
    │   ├── diffusion_head/    # 扩散头组件
    │   └── samplers/          # 采样器
    ├── modules/               # 自定义模块
    │   ├── attention.py       # 注意力机制
    │   ├── encoders/          # 编码器
    │   ├── networks/          # 网络组件
    │   └── vision/            # 视觉组件
    └── utils/                 # 工具函数
```

### 核心组件

#### 1. 数据层 (`src/unifolm_wma/data/`)
- **WMAData类** (`wma_data.py`): 主要数据集类，处理多模态数据
  - 支持视频、机器人状态、动作数据
  - 数据格式：H5文件存储状态/动作，MP4存储视频
  - 标准化和反标准化处理

#### 2. 模型层 (`src/unifolm_wma/models/`)
- **DDPM** (`ddpms.py`): 去噪扩散概率模型，继承自PyTorch Lightning
  - 支持两种模式：决策制定模式和仿真模式
  - 集成视觉编码器和动作头
  - 使用DDIM采样器进行推理

- **扩散头** (`diffusion_head/`): 处理动作预测的专用网络
  - 条件UNet1D架构
  - 支持位置编码
  - EMA模型用于稳定训练

#### 3. 模块层 (`src/unifolm_wma/modules/`)
- **注意力机制** (`attention.py`): 多种注意力实现
  - CrossAttention, SpatialTransformer, TemporalTransformer
  - 支持xformers优化
  - 相对位置编码

- **视觉组件** (`vision/`): 视觉编码器
  - DINO-SigLIP ViT架构
  - 基础视觉模块

### 配置系统

#### 训练配置 (`configs/train/config.yaml`)
```yaml
model:
  target: unifolm_wma.models.ddpms.LatentVisualDiffusion
  params:
    # 扩散模型参数
    timesteps: 1000
    linear_start: 0.00085
    linear_end: 0.012
    
    # 输入维度配置
    agent_state_dim: 16      # 机器人状态维度 (最大16自由度)
    agent_action_dim: 16     # 机器人动作维度
    
    # 模式选择
    decision_making_only: true  # 仅决策模式 vs 联合训练
    
    # 观测步数
    n_obs_steps_imagen: 2    # 图像观测步数
    n_obs_steps_acting: 2    # 动作观测步数
```

#### 数据元信息 (`configs/train/meta.json`)
```json
{
    "obs": {
        "image": {"shape": [3, 320, 512], "type": "rgb"},
        "agent_pos": {"shape": [16], "type": "low_dim"}
    },
    "action": {"shape": [16]}
}
```

## 典型使用流程

### 1. 训练流程

#### A. 数据准备
```bash
# 预处理训练数据
python prepare_data/prepare_training_data.py \
    --source_dir /path/to/raw/data \
    --target_dir /path/to/processed/data \
    --dataset_name your_dataset
```

数据结构要求：
```
dataset_dir/
├── videos/
│   └── dataset_name/
│       └── camera_view/
│           ├── 0.mp4
│           └── 1.mp4
├── transitions/
│   └── dataset_name/
│       ├── 0.h5          # 包含状态和动作数据
│       └── 1.h5
└── dataset_name.csv      # 元数据索引
```

#### B. 配置设置
1. 更新 `configs/train/config.yaml` 中的模型参数
2. 设置预训练检查点路径
3. 配置数据目录和数据集权重

#### C. 训练执行
```bash
# 单GPU训练
python scripts/trainer.py \
    --base configs/train/config.yaml \
    --name experiment_name \
    --train

# 多GPU分布式训练
torchrun --nproc_per_node=8 scripts/trainer.py \
    --base configs/train/config.yaml \
    --name experiment_name \
    --train
```

#### 训练策略（三步骤）：
1. **步骤1**: 在Open-X数据集上微调视频生成模型作为世界模型
2. **步骤2**: 在下游任务数据集上进行决策模式后训练
3. **步骤3**: 在下游任务数据集上进行仿真模式后训练

### 2. 推理流程

#### A. 基础模型推理
```bash
python scripts/evaluation/base_model_inference.py \
    --config configs/inference/base_model_inference.yaml \
    --ckpt_path /path/to/checkpoint \
    --prompt_dir /path/to/prompts \
    --savedir /path/to/output
```

#### B. 世界模型交互推理
```bash
python scripts/evaluation/world_model_interaction.py \
    --config configs/inference/world_model_interaction.yaml \
    --ckpt_path /path/to/checkpoint \
    --prompt_dir /path/to/prompts \
    --dataset dataset_name \
    --savedir /path/to/output
```

### 3. 核心工作流程

#### 训练阶段流程：
1. **数据加载**: WMAData类加载多模态数据（视频、状态、动作）
2. **模型初始化**: 实例化LatentVisualDiffusion模型
3. **前向传播**: 
   - 编码输入图像到潜在空间
   - 处理语言指令和机器人状态
   - 扩散过程预测未来帧和动作
4. **损失计算**: 计算扩散损失和动作预测损失
5. **反向传播**: 使用AdamW优化器更新参数

#### 推理阶段流程：
1. **模型加载**: 加载预训练检查点
2. **输入处理**: 处理观测图像、语言指令、机器人状态
3. **DDIM采样**: 
   - 使用DDIM采样器生成未来帧
   - 同时预测机器人动作序列
4. **后处理**: 将潜在表示解码为像素空间
5. **输出**: 生成的视频帧和动作序列

### 4. 关键技术特性

#### 多模态融合：
- **视觉**: RGB图像序列，支持多视角
- **语言**: 自然语言指令编码
- **状态**: 机器人关节位置和速度
- **动作**: 机器人控制指令

#### 双模式架构：
- **决策模式**: 直接预测最优动作
- **仿真模式**: 预测状态转换和未来观测

#### 扩散架构：
- 基于DDPM的生成模型
- 支持条件生成和无条件生成
- 集成视觉-语言-动作的联合建模

### 5. 模型检查点

| 模型 | 描述 | 用途 |
|------|------|------|
| UnifoLM-WMA-0_Base | 在Open-X数据集上微调 | 基础世界模型 |
| UnifoLM-WMA-0_Dual | 在Unitree数据集上双模式训练 | 生产就绪模型 |

### 6. 依赖要求

核心依赖：
- Python 3.10.18
- PyTorch 2.3.1 + torchvision 0.18.1
- PyTorch Lightning 1.9.3
- xformers 0.0.27 (注意力优化)
- transformers 4.40.1
- 其他：einops, omegaconf, pandas等

## 总结

UnifoLM-WMA是一个完整的端到端机器人学习框架，集成了最新的扩散模型技术、多模态学习和世界模型方法。其模块化设计支持灵活配置，适用于各种机器人应用场景。通过世界模型的双重功能（仿真和策略增强），该框架能够有效利用数据并提升机器人的决策性能。