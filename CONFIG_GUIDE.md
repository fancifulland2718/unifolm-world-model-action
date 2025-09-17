# UnifoLM-WMA 配置指南

## 1. 环境配置

### 基本环境安装
```bash
# 创建conda环境
conda create -n unifolm-wma python==3.10.18
conda activate unifolm-wma

# 安装系统依赖
conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge

# 克隆项目 (包含子模块)
git clone --recurse-submodules https://github.com/unitreerobotics/unifolm-world-model-action.git
cd unifolm-world-model-action

# 如果已下载项目，初始化子模块
git submodule update --init --recursive

# 安装项目依赖
pip install -e .

# 安装外部依赖
cd external/dlimp
pip install -e .
cd ../..
```

### GPU要求
- CUDA 11.8+ 或 CUDA 12.x
- 至少8GB显存 (推荐16GB+)
- 支持多GPU训练

## 2. 数据配置

### 数据格式要求
```
dataset_dir/
├── videos/                     # 视频数据
│   └── dataset_name/
│       └── camera_view/
│           ├── 0.mp4          # 第一个轨迹的视频
│           ├── 1.mp4          # 第二个轨迹的视频
│           └── ...
├── transitions/               # 状态和动作数据
│   └── dataset_name/
│       ├── meta_data/         # 元数据
│       ├── 0.h5              # 第一个轨迹的状态/动作
│       ├── 1.h5              # 第二个轨迹的状态/动作
│       └── ...
└── dataset_name.csv          # 索引文件
```

### H5文件格式
每个H5文件应包含:
```python
{
    'observation': {
        'agent_pos': np.array(...),    # shape: [T, state_dim]
        'image': np.array(...),        # shape: [T, H, W, 3] (可选，如果有的话)
    },
    'action': np.array(...),           # shape: [T, action_dim]
    'language': string                 # 语言指令
}
```

### CSV索引文件格式
```csv
videoid,contentUrl,duration,data_dir,name,instruction,width,height,fps,frame_num,...
0,videos/dataset_name/camera_view/0.mp4,10.0,transitions/dataset_name,0,pick up the object,512,320,30,300,...
1,videos/dataset_name/camera_view/1.mp4,8.5,transitions/dataset_name,1,place the object,512,320,30,255,...
```

## 3. 训练配置

### 主配置文件: `configs/train/config.yaml`

#### 模型参数配置
```yaml
model:
  pretrained_checkpoint: /path/to/unifolm-wma-base.ckpt  # 必须: 预训练检查点路径
  base_learning_rate: 1.0e-05                           # 学习率
  target: unifolm_wma.models.ddpms.LatentVisualDiffusion
  params:
    # === 核心参数 ===
    agent_state_dim: 16        # 机器人状态维度 (根据你的机器人调整)
    agent_action_dim: 16       # 机器人动作维度 (根据你的机器人调整)
    decision_making_only: true # true: 仅决策模式, false: 联合训练两种模式
    
    # === 观测参数 ===
    n_obs_steps_imagen: 2      # 图像观测历史步数
    n_obs_steps_acting: 2      # 动作观测历史步数
    
    # === 扩散模型参数 ===
    timesteps: 1000            # 扩散步数
    linear_start: 0.00085      # 噪声调度起始值
    linear_end: 0.012          # 噪声调度结束值
    
    # === 图像参数 ===
    image_size: [40, 64]       # 潜在空间图像尺寸 (实际图像尺寸的1/8)
    channels: 4                # 潜在空间通道数
    
    # === 训练参数 ===
    use_ema: false             # 是否使用指数移动平均
    uncond_prob: 0.05          # 无条件训练概率
    perframe_ae: true          # 逐帧自编码器
```

#### 数据配置
```yaml
data:
  target: unifolm_wma.data.base.DataModuleFromConfig
  params:
    batch_size: 1              # 批大小 (通常设为1)
    num_workers: 8             # 数据加载线程数
    train:
      target: unifolm_wma.data.wma_data.WMAData
      params:
        meta_path: configs/train/meta.json
        data_dir: /path/to/your/dataset   # 你的数据集路径
        subsample: null          # 数据子采样 (null表示使用全部)
        video_length: 16         # 视频长度
        resolution: [320, 512]   # 图像分辨率 [H, W]
        
    # 数据集权重配置 (权重和必须为1.0)
    dataset_and_weights:
      your_dataset_name: 1.0
```

### 元数据配置: `configs/train/meta.json`
```json
{
    "obs": {
        "image": {
            "shape": [3, 320, 512],    # 图像形状 [C, H, W]
            "type": "rgb"
        },
        "agent_pos": {
            "shape": [16],             # 状态维度 (与agent_state_dim一致)
            "type": "low_dim"
        }
    },
    "action": {
        "shape": [16]                  # 动作维度 (与agent_action_dim一致)
    }
}
```

## 4. 推理配置

### 世界模型交互推理: `configs/inference/world_model_interaction.yaml`
```yaml
model:
  target: unifolm_wma.models.ddpms.LatentVisualDiffusion
  params:
    # 与训练配置保持一致
    agent_state_dim: 16
    agent_action_dim: 16
    # ... 其他参数

# 推理特定参数
inference:
  ddim_steps: 50               # DDIM采样步数
  ddim_eta: 0.0               # DDIM eta参数
  guidance_scale: 7.5         # 引导强度
  height: 320                 # 输出图像高度
  width: 512                  # 输出图像宽度
```

## 5. 常见配置调整

### 适配不同机器人
```yaml
# 7自由度机械臂
agent_state_dim: 7
agent_action_dim: 7

# 移动机器人 (x, y, theta)
agent_state_dim: 3
agent_action_dim: 3

# 双臂机器人
agent_state_dim: 14    # 7DOF × 2
agent_action_dim: 14
```

### 内存优化
```yaml
# 减少批大小和视频长度
batch_size: 1
video_length: 8

# 减少工作线程
num_workers: 4

# 使用梯度累积
gradient_accumulate_every: 2
```

### 性能优化
```yaml
# 启用混合精度
precision: 16

# 使用更大批大小 (如果内存允许)
batch_size: 2

# 增加工作线程
num_workers: 16
```

## 6. 常见问题解决

### 内存不足
1. 减少 `batch_size` 和 `video_length`
2. 减少 `num_workers`
3. 使用 `precision: 16`
4. 启用 `gradient_accumulate_every`

### 训练不稳定
1. 调整学习率: `base_learning_rate: 5.0e-06`
2. 启用EMA: `use_ema: true`
3. 调整噪声调度: `linear_start: 0.0001, linear_end: 0.02`

### 推理速度慢
1. 减少DDIM步数: `ddim_steps: 25`
2. 使用更小的图像尺寸
3. 启用xformers优化

### 数据加载错误
1. 检查数据路径和格式
2. 验证H5文件结构
3. 确认CSV索引文件格式
4. 检查视频文件可读性

## 7. 监控和调试

### TensorBoard监控
```bash
tensorboard --logdir logs/experiment_name/tensorboard
```

### 日志位置
```
logs/
└── experiment_name/
    ├── tensorboard/        # TensorBoard日志
    ├── checkpoints/        # 模型检查点
    ├── configs/           # 保存的配置
    └── log_files/         # 训练日志
```

### 调试技巧
1. 使用小数据集测试配置
2. 检查数据加载器输出
3. 监控损失曲线
4. 定期保存检查点
5. 使用梯度裁剪防止梯度爆炸