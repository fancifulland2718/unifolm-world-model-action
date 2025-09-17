# UnifoLM-WMA 快速入门指南

## 🚀 5分钟快速上手

### 1. 快速安装
```bash
# 创建环境并安装
conda create -n unifolm-wma python==3.10.18
conda activate unifolm-wma
conda install pinocchio=3.2.0 ffmpeg=7.1.1 -c conda-forge -y

# 克隆并安装项目
git clone --recurse-submodules https://github.com/unitreerobotics/unifolm-world-model-action.git
cd unifolm-world-model-action
pip install -e .
cd external/dlimp && pip install -e . && cd ../..
```

### 2. 快速体验 - 查看架构信息
```bash
python examples/usage_example.py --mode info
```

### 3. 下载预训练模型
```bash
# 下载 UnifoLM-WMA-0_Base 模型
# 从 HuggingFace: https://huggingface.co/unitreerobotics/UnifoLM-WMA-0
# 保存到: models/unifolm-wma-base.ckpt
```

### 4. 准备你的数据 (可选 - 使用自己的数据)
```bash
python examples/usage_example.py --mode prepare_data \
    --source_dir /path/to/your/raw/data \
    --target_dir /path/to/processed/data \
    --dataset_name my_robot_task
```

### 5. 开始训练
```bash
# 编辑配置文件
# 1. 更新 configs/train/config.yaml 中的数据路径
# 2. 设置预训练检查点路径
# 3. 根据你的机器人调整 agent_state_dim 和 agent_action_dim

# 开始训练
python examples/usage_example.py --mode train \
    --config configs/train/config.yaml \
    --name my_first_experiment
```

### 6. 运行推理
```bash
python examples/usage_example.py --mode inference \
    --ckpt_path /path/to/your/trained/model.ckpt \
    --prompt_dir examples/ \
    --dataset my_robot_task
```

## 📊 典型训练时间和资源需求

| 场景 | GPU | 训练时间 | 显存需求 |
|------|-----|----------|----------|
| 小型数据集 (1K轨迹) | 1×RTX 3090 | 2-4小时 | 12GB |
| 中型数据集 (10K轨迹) | 2×RTX 4090 | 8-12小时 | 24GB×2 |
| 大型数据集 (100K轨迹) | 8×A100 | 1-2天 | 40GB×8 |

## 🎯 常见使用场景

### 场景1: 机械臂抓取任务
```yaml
# configs/train/config.yaml
model:
  params:
    agent_state_dim: 7     # 7自由度机械臂
    agent_action_dim: 7    # 关节角度控制
    decision_making_only: true
```

### 场景2: 移动机器人导航
```yaml
# configs/train/config.yaml  
model:
  params:
    agent_state_dim: 3     # x, y, theta
    agent_action_dim: 2    # 线速度, 角速度
    decision_making_only: true
```

### 场景3: 双臂协作任务
```yaml
# configs/train/config.yaml
model:
  params:
    agent_state_dim: 14    # 7DOF × 2臂
    agent_action_dim: 14   # 双臂联合控制
    decision_making_only: false  # 启用仿真模式
```

## 🔧 快速配置检查清单

训练前检查以下配置:

### ✅ 数据配置
- [ ] 数据路径正确: `data.params.train.params.data_dir`
- [ ] 数据集名称匹配: `data.params.dataset_and_weights`
- [ ] meta.json 维度正确: `obs.agent_pos.shape` 和 `action.shape`

### ✅ 模型配置  
- [ ] 预训练检查点存在: `model.pretrained_checkpoint`
- [ ] 状态维度正确: `model.params.agent_state_dim`
- [ ] 动作维度正确: `model.params.agent_action_dim`
- [ ] 图像尺寸合理: `data.params.train.params.resolution`

### ✅ 资源配置
- [ ] GPU内存足够 (至少8GB)
- [ ] 存储空间足够 (模型+日志+检查点)
- [ ] CPU和内存适当: `data.params.num_workers`

## 🐛 常见问题快速解决

### 问题1: 内存不足 (CUDA out of memory)
```yaml
# 解决方案: 减少批大小和视频长度
data:
  params:
    batch_size: 1
    train:
      params:
        video_length: 8
        num_workers: 4
```

### 问题2: 数据加载错误
```bash
# 检查数据完整性
python -c "
import h5py
f = h5py.File('transitions/dataset_name/0.h5', 'r')
print('Keys:', list(f.keys()))
print('Observation keys:', list(f['observation'].keys()))
print('Action shape:', f['action'].shape)
"
```

### 问题3: 训练损失不下降
```yaml
# 解决方案: 调整学习率和预处理
model:
  base_learning_rate: 5.0e-06  # 降低学习率
  params:
    uncond_prob: 0.1            # 增加无条件训练
    input_pertub: 0.05          # 减少输入扰动
```

### 问题4: 推理结果质量差
```yaml
# 解决方案: 调整推理参数
inference:
  ddim_steps: 100              # 增加采样步数
  guidance_scale: 10.0         # 增加引导强度
  ddim_eta: 0.1               # 调整随机性
```

## 📈 性能优化建议

### 训练加速
1. **使用混合精度**: 在配置中设置 `precision: 16`
2. **多GPU训练**: 使用 `torchrun --nproc_per_node=N`
3. **数据预加载**: 增加 `num_workers` 数量
4. **梯度累积**: 设置 `gradient_accumulate_every: 2`

### 推理加速
1. **减少DDIM步数**: `ddim_steps: 25-50`
2. **批量推理**: 同时处理多个样本
3. **模型量化**: 使用torch.jit或TensorRT
4. **缓存优化**: 预加载模型权重

## 🎓 进阶学习路径

1. **理解架构**: 阅读 `ARCHITECTURE_ANALYSIS.md`
2. **深入配置**: 学习 `CONFIG_GUIDE.md`  
3. **查看源码**: 从 `src/unifolm_wma/models/ddpms.py` 开始
4. **实验对比**: 尝试不同的模型配置和训练策略
5. **社区交流**: 参与GitHub Issues和Discussions

## 📞 获取帮助

- **文档**: 查看README和配置指南
- **示例**: 运行 `examples/usage_example.py`
- **Issues**: 在GitHub上提交问题
- **论文**: 阅读相关技术论文了解原理

---

🎉 **恭喜!** 你已经掌握了UnifoLM-WMA的基本使用方法。现在开始构建你的机器人世界模型吧!