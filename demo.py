"""
MMEA 模型演示脚本
功能：加载训练好的模型，展示推理结果

使用方法：
    # 方式1：使用默认参数
    python demo.py
    
    # 方式2：指定模型和数据集
    python demo.py --model_path pkl/model_epoch_999_dbp15k_zh_en.pkl --file_dir data/DBP15K/zh_en --cuda
"""

import os
import sys
import torch
import numpy as np
import argparse

# 添加 src 到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from run import MCLEA
from utils import pairwise_distances


def demo_inference(model_path, file_dir, cuda=False):
    """
    演示模型推理过程
    
    Args:
        model_path: 模型文件路径
        file_dir: 数据集路径
        cuda: 是否使用GPU
    """
    print("=" * 60)
    print("MMEA Model Demo - Inference Demonstration")
    print("=" * 60)
    
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    
    # ========== 1. 加载模型 ==========
    print("\n" + "-" * 60)
    print("[Step 1] Loading trained model...")
    print("-" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print(f"Please train the model first or check the path.")
        return
    
    # 加载保存的字典
    save_dict = torch.load(model_path, map_location=device)
    saved_args = save_dict['args']
    
    # 将保存的参数转换为命令行参数
    import sys
    original_argv = sys.argv[:]
    
    # 构建命令行参数
    sys.argv = ['demo.py']
    for key, value in saved_args.items():
        if isinstance(value, bool):
            if value:
                sys.argv.append(f'--{key}')
        else:
            sys.argv.append(f'--{key}={value}')
    
    # 初始化模型（现在会使用正确的参数）
    model = MCLEA()
    
    # 恢复原始 argv
    sys.argv = original_argv
    
    # 加载保存的参数
    print("Loading parameters...")
    model.multimodal_encoder.load_state_dict(save_dict['multimodal_encoder'])
    model.multi_loss_layer.load_state_dict(save_dict['multi_loss_layer'])
    model.align_multi_loss_layer.load_state_dict(save_dict['align_multi_loss_layer'])
    
    # 设置为评估模式
    model.multimodal_encoder.eval()
    print("✓ Model initialized successfully")
    
    # 显示模型结构信息
    print(f"\n  - Entity number: {model.ENT_NUM}")
    print(f"  - Relation number: {model.REL_NUM}")
    print(f"  - Training samples: {model.train_ill.shape[0]}")
    print(f"  - Test samples: {model.test_ill.shape[0]}")
    
    # ========== 3. 模型推理 ==========
    print("\n" + "-" * 60)
    print("[Step 3] Running inference...")
    print("-" * 60)
    
    with torch.no_grad():
        # 前向传播
        *embs, _ = model.multimodal_encoder(
            model.input_idx,
            model.adj,
            model.img_features,
            model.rel_features,
            model.att_features,
            model.name_features,
            model.char_features
        )
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = embs[:7]
        
        # 归一化
        final_emb = torch.nn.functional.normalize(joint_emb)
        
        print(f"✓ Embeddings computed successfully")
        print(f"  - Joint embedding shape: {joint_emb.shape}")
        print(f"  - Final embedding shape: {final_emb.shape}")
        
        # ========== 4. 计算评估指标 ==========
        print("\n" + "-" * 60)
        print("[Step 4] Computing evaluation metrics...")
        print("-" * 60)
        
        test_left = model.test_left
        test_right = model.test_right
        
        # 计算距离矩阵
        distance = pairwise_distances(final_emb[test_left], final_emb[test_right])
        
        # 计算指标
        top_k = [1, 10, 50]
        acc_l2r = np.zeros(len(top_k), dtype=np.float32)
        acc_r2l = np.zeros(len(top_k), dtype=np.float32)
        mrr_l2r = 0.0
        mrr_r2l = 0.0
        
        # L2R
        for idx in range(test_left.shape[0]):
            values, indices = torch.sort(distance[idx, :], descending=False)
            rank = (indices == idx).nonzero().squeeze().item()
            mrr_l2r += 1.0 / (rank + 1)
            for i, k in enumerate(top_k):
                if rank < k:
                    acc_l2r[i] += 1
        
        # R2L
        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=False)
            rank = (indices == idx).nonzero().squeeze().item()
            mrr_r2l += 1.0 / (rank + 1)
            for i, k in enumerate(top_k):
                if rank < k:
                    acc_r2l[i] += 1
        
        # 归一化
        acc_l2r = acc_l2r / test_left.shape[0]
        acc_r2l = acc_r2l / test_right.shape[0]
        mrr_l2r = mrr_l2r / test_left.shape[0]
        mrr_r2l = mrr_r2l / test_right.shape[0]
        
        # ========== 5. 展示结果 ==========
        print("\n" + "=" * 60)
        print("📊 Evaluation Results")
        print("=" * 60)
        print(f"\n{'Metric':<15} {'L2R':<15} {'R2L':<15}")
        print("-" * 60)
        print(f"{'Hits@1':<15} {acc_l2r[0]:<15.4f} {acc_r2l[0]:<15.4f}")
        print(f"{'Hits@10':<15} {acc_l2r[1]:<15.4f} {acc_r2l[1]:<15.4f}")
        print(f"{'Hits@50':<15} {acc_l2r[2]:<15.4f} {acc_r2l[2]:<15.4f}")
        print(f"{'MRR':<15} {mrr_l2r:<15.4f} {mrr_r2l:<15.4f}")
        print("=" * 60)
        
        # ========== 6. 示例：展示几个具体预测 ==========
        print("\n" + "-" * 60)
        print("[Step 5] Example predictions (Top-3 for first 5 queries)")
        print("-" * 60)
        
        for i in range(min(5, test_left.shape[0])):
            query_idx = test_left[i].item()
            _, indices = torch.sort(distance[i, :], descending=False)
            top3_indices = indices[:3].cpu().numpy()
            top3_entities = [test_right[idx].item() for idx in top3_indices]
            
            # 检查是否预测正确
            gt_idx = test_right[i].item()
            is_correct = gt_idx in top3_entities
            
            print(f"\nQuery entity: {query_idx}")
            print(f"  Ground truth: {gt_idx} {'✓' if is_correct else '✗'}")
            print(f"  Top-3 predictions: {top3_entities}")
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("=" * 60)
        
        return {
            'hits@1': acc_l2r[0],
            'hits@10': acc_l2r[1],
            'mrr': mrr_l2r
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMEA Model Demo Script")
    
    # 自动查找模型文件
    default_model = None
    pkl_dir = os.path.join(os.path.dirname(__file__), "pkl")
    if os.path.exists(pkl_dir):
        model_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
        if model_files:
            # 优先选择 best_model，否则选择第一个
            best_models = [f for f in model_files if f.startswith('best_model')]
            if best_models:
                default_model = os.path.join(pkl_dir, best_models[0])
            else:
                default_model = os.path.join(pkl_dir, model_files[0])
    
    parser.add_argument("--model_path", type=str, default=default_model,
                        help="模型文件路径 (默认自动查找 pkl/ 目录)")
    parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en",
                        help="数据集路径")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="是否使用GPU")
    
    args = parser.parse_args()
    
    if args.model_path is None:
        print("❌ Error: No model file found in pkl/ directory.")
        print("Please train the model first by running:")
        print("  python src/run.py --file_dir data/DBP15K/zh_en --cuda")
        sys.exit(1)
    
    print(f"Using model: {args.model_path}")
    
    # 运行演示
    results = demo_inference(
        model_path=args.model_path,
        file_dir=args.file_dir,
        cuda=args.cuda
    )
