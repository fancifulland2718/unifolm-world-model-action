#!/usr/bin/env python3
"""
UnifoLM-WMA 使用示例脚本
演示典型的训练和推理流程

Usage Examples:
1. 数据准备:
   python examples/usage_example.py --mode prepare_data --source_dir /path/to/raw --target_dir /path/to/processed

2. 训练:
   python examples/usage_example.py --mode train --config configs/train/config.yaml --name my_experiment

3. 推理:
   python examples/usage_example.py --mode inference --ckpt_path /path/to/checkpoint --prompt_dir /path/to/prompts
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def prepare_data_example(args):
    """数据准备示例"""
    print("=== 数据准备示例 ===")
    
    cmd = [
        "python", "prepare_data/prepare_training_data.py",
        "--source_dir", args.source_dir,
        "--target_dir", args.target_dir,
        "--dataset_name", args.dataset_name or "example_dataset"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 检查源目录是否存在
    if not os.path.exists(args.source_dir):
        print(f"错误: 源目录不存在: {args.source_dir}")
        return False
    
    # 创建目标目录
    os.makedirs(args.target_dir, exist_ok=True)
    
    try:
        subprocess.run(cmd, check=True)
        print("数据准备完成!")
        
        # 显示预期的数据结构
        print(f"\n生成的数据结构应该如下:")
        print(f"{args.target_dir}/")
        print(f"├── videos/")
        print(f"│   └── {args.dataset_name or 'example_dataset'}/")
        print(f"│       └── camera_view/")
        print(f"│           ├── 0.mp4")
        print(f"│           └── 1.mp4")
        print(f"├── transitions/")
        print(f"│   └── {args.dataset_name or 'example_dataset'}/")
        print(f"│       ├── 0.h5")
        print(f"│       └── 1.h5")
        print(f"└── {args.dataset_name or 'example_dataset'}.csv")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"数据准备失败: {e}")
        return False


def train_example(args):
    """训练示例"""
    print("=== 训练示例 ===")
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return False
    
    # 基础训练命令
    cmd = [
        "python", "scripts/trainer.py",
        "--base", args.config,
        "--name", args.name,
        "--train"
    ]
    
    # 添加可选参数
    if args.gpus:
        cmd.extend(["--gpus", str(args.gpus)])
    if args.logdir:
        cmd.extend(["--logdir", args.logdir])
    
    print(f"执行训练命令: {' '.join(cmd)}")
    
    # 显示训练配置信息
    print(f"\n训练配置:")
    print(f"- 配置文件: {args.config}")
    print(f"- 实验名称: {args.name}")
    print(f"- GPU数量: {args.gpus or '自动检测'}")
    print(f"- 日志目录: {args.logdir or 'logs'}")
    
    try:
        subprocess.run(cmd, check=True)
        print("训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False


def inference_example(args):
    """推理示例"""
    print("=== 推理示例 ===")
    
    # 检查检查点文件
    if not os.path.exists(args.ckpt_path):
        print(f"错误: 检查点文件不存在: {args.ckpt_path}")
        return False
    
    # 检查提示目录
    if not os.path.exists(args.prompt_dir):
        print(f"错误: 提示目录不存在: {args.prompt_dir}")
        return False
    
    # 推理配置
    inference_config = args.inference_config or "configs/inference/world_model_interaction.yaml"
    
    # 推理命令
    cmd = [
        "python", "scripts/evaluation/world_model_interaction.py",
        "--config", inference_config,
        "--ckpt_path", args.ckpt_path,
        "--prompt_dir", args.prompt_dir,
        "--savedir", args.savedir or "output"
    ]
    
    # 添加可选参数
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.height:
        cmd.extend(["--height", str(args.height)])
    if args.width:
        cmd.extend(["--width", str(args.width)])
    
    print(f"执行推理命令: {' '.join(cmd)}")
    
    # 显示推理配置信息
    print(f"\n推理配置:")
    print(f"- 配置文件: {inference_config}")
    print(f"- 检查点: {args.ckpt_path}")
    print(f"- 提示目录: {args.prompt_dir}")
    print(f"- 输出目录: {args.savedir or 'output'}")
    print(f"- 数据集: {args.dataset or '自动检测'}")
    print(f"- 图像尺寸: {args.height or 320}x{args.width or 512}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"推理完成! 结果保存在: {args.savedir or 'output'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"推理失败: {e}")
        return False


def show_architecture_info():
    """显示架构信息"""
    print("=== UnifoLM-WMA 架构概述 ===")
    print("""
核心组件:
1. 数据层 (src/unifolm_wma/data/)
   - WMAData: 多模态数据加载 (视频、状态、动作)
   - 支持H5格式的状态/动作数据, MP4格式的视频数据

2. 模型层 (src/unifolm_wma/models/)
   - DDPM: 去噪扩散概率模型 (基于PyTorch Lightning)
   - 支持决策模式和仿真模式
   - 集成视觉编码器和动作头

3. 模块层 (src/unifolm_wma/modules/)
   - 注意力机制: CrossAttention, SpatialTransformer等
   - 视觉组件: DINO-SigLIP ViT架构

训练流程:
1. 数据预处理: 将原始数据转换为标准格式
2. 模型配置: 设置模型参数和训练超参数
3. 三阶段训练:
   - 步骤1: Open-X数据集上的世界模型微调
   - 步骤2: 下游任务的决策模式后训练
   - 步骤3: 下游任务的仿真模式后训练

推理流程:
1. 模型加载: 加载预训练检查点
2. 输入处理: 处理多模态输入 (图像、语言、状态)
3. DDIM采样: 生成未来帧和动作序列
4. 后处理: 解码和输出结果

支持的模态:
- 视觉: RGB图像序列 (多视角)
- 语言: 自然语言指令
- 状态: 机器人关节位置/速度 (最大16自由度)
- 动作: 机器人控制指令
    """)


def main():
    parser = argparse.ArgumentParser(description="UnifoLM-WMA使用示例")
    parser.add_argument("--mode", choices=["prepare_data", "train", "inference", "info"], 
                       required=True, help="运行模式")
    
    # 数据准备参数
    parser.add_argument("--source_dir", help="原始数据目录")
    parser.add_argument("--target_dir", help="目标数据目录")
    parser.add_argument("--dataset_name", default="example_dataset", help="数据集名称")
    
    # 训练参数
    parser.add_argument("--config", default="configs/train/config.yaml", help="训练配置文件")
    parser.add_argument("--name", default="example_experiment", help="实验名称")
    parser.add_argument("--gpus", type=int, help="GPU数量")
    parser.add_argument("--logdir", help="日志目录")
    
    # 推理参数
    parser.add_argument("--ckpt_path", help="检查点路径")
    parser.add_argument("--prompt_dir", help="提示目录")
    parser.add_argument("--savedir", default="output", help="输出目录")
    parser.add_argument("--inference_config", help="推理配置文件")
    parser.add_argument("--dataset", help="数据集名称")
    parser.add_argument("--height", type=int, default=320, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    
    args = parser.parse_args()
    
    # 检查当前目录是否为项目根目录
    if not os.path.exists("src/unifolm_wma"):
        print("错误: 请在项目根目录下运行此脚本")
        print("当前目录应包含 src/unifolm_wma/ 文件夹")
        sys.exit(1)
    
    success = False
    
    if args.mode == "prepare_data":
        if not args.source_dir or not args.target_dir:
            print("错误: 数据准备模式需要 --source_dir 和 --target_dir 参数")
            sys.exit(1)
        success = prepare_data_example(args)
        
    elif args.mode == "train":
        success = train_example(args)
        
    elif args.mode == "inference":
        if not args.ckpt_path or not args.prompt_dir:
            print("错误: 推理模式需要 --ckpt_path 和 --prompt_dir 参数")
            sys.exit(1)
        success = inference_example(args)
        
    elif args.mode == "info":
        show_architecture_info()
        success = True
    
    if success:
        print(f"\n✅ {args.mode} 模式执行成功!")
    else:
        print(f"\n❌ {args.mode} 模式执行失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()