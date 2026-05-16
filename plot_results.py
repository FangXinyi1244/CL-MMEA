import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适合服务器环境
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 读取数据
def load_and_clean_csv(file_path):
    """读取CSV文件并去除重复数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('epoch,'):
            continue
        # 去掉行号前缀
        if ':' in line:
            line = line.split(':', 1)[1]
        parts = line.split(',')
        if len(parts) > 1:
            try:
                row = [float(x) for x in parts]
                data.append(row)
            except:
                continue
    
    df = pd.DataFrame(data)
    if len(df) > 0:
        df = df.drop_duplicates(subset=[0])
        df = df.sort_values(by=0)
    return df

# 文件路径
base_dir = r'g:\PersonalUse\learningSource\毕业设计\working\MCLEA\CL-MMEA'
original_dir = os.path.join(base_dir, 'save_pkl\原方案')
improved_dir = os.path.join(base_dir, 'save_pkl\改进方案1')

# 读取数据
original_train = load_and_clean_csv(os.path.join(original_dir, 'baseline_train_loss.csv'))
original_test = load_and_clean_csv(os.path.join(original_dir, 'baseline_test_log.csv'))
improved_train = load_and_clean_csv(os.path.join(improved_dir, 'baseline_train_loss.csv'))
improved_test = load_and_clean_csv(os.path.join(improved_dir, 'baseline_test_log.csv'))
improved_mask = load_and_clean_csv(os.path.join(improved_dir, 'masked_train_loss.csv'))

print("数据加载完成，开始绘制图表...")

# 图表尺寸
fig_size = (10, 6)

# 图1：训练总损失对比
plt.figure(figsize=fig_size)
if len(original_train) > 0:
    plt.plot(original_train[0], original_train[4], label='原方案', linewidth=2)
if len(improved_train) > 0:
    plt.plot(improved_train[0], improved_train[4], label='改进方案1 (掩码特征)', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Total Loss', fontsize=12)
plt.title('训练总损失对比', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'save_pkl', '训练损失对比.png'), dpi=300)
plt.close()

# 图2：Hits@1 对比
plt.figure(figsize=fig_size)
if len(original_test) > 0:
    plt.plot(original_test[0], original_test[1], label='原方案', linewidth=2, marker='o', markersize=4)
if len(improved_test) > 0:
    plt.plot(improved_test[0], improved_test[1], label='改进方案1 (掩码特征)', linewidth=2, marker='s', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Hits@1', fontsize=12)
plt.title('Hits@1 对比', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'save_pkl', 'Hits@1对比.png'), dpi=300)
plt.close()

# 图3：Hits@10 对比
plt.figure(figsize=fig_size)
if len(original_test) > 0:
    plt.plot(original_test[0], original_test[2], label='原方案', linewidth=2, marker='o', markersize=4)
if len(improved_test) > 0:
    plt.plot(improved_test[0], improved_test[2], label='改进方案1 (掩码特征)', linewidth=2, marker='s', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Hits@10', fontsize=12)
plt.title('Hits@10 对比', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'save_pkl', 'Hits@10对比.png'), dpi=300)
plt.close()

# 图4：MRR 对比
plt.figure(figsize=fig_size)
if len(original_test) > 0:
    plt.plot(original_test[0], original_test[3], label='原方案', linewidth=2, marker='o', markersize=4)
if len(improved_test) > 0:
    plt.plot(improved_test[0], improved_test[3], label='改进方案1 (掩码特征)', linewidth=2, marker='s', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MRR', fontsize=12)
plt.title('MRR 对比', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'save_pkl', 'MRR对比.png'), dpi=300)
plt.close()

# 图5：距离指标对比
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
if len(original_test) > 0:
    axes[0].plot(original_test[0], original_test[4], label='原方案', linewidth=2)
    axes[1].plot(original_test[0], original_test[5], label='原方案', linewidth=2)
    axes[2].plot(original_test[0], original_test[6], label='原方案', linewidth=2)
if len(improved_test) > 0:
    axes[0].plot(improved_test[0], improved_test[4], label='改进方案1', linewidth=2)
    axes[1].plot(improved_test[0], improved_test[5], label='改进方案1', linewidth=2)
    axes[2].plot(improved_test[0], improved_test[6], label='改进方案1', linewidth=2)

axes[0].set_ylabel('Pos Distance', fontsize=11)
axes[0].set_title('正样本距离', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel('Neg Distance', fontsize=11)
axes[1].set_title('负样本距离', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Margin', fontsize=11)
axes[2].set_title('间隔 (Margin)', fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'save_pkl', '距离指标对比.png'), dpi=300)
plt.close()

# 图6：掩码损失曲线
if len(improved_mask) > 0:
    plt.figure(figsize=fig_size)
    plt.plot(improved_mask[0], improved_mask[1], label='掩码对比损失', linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mask Loss', fontsize=12)
    plt.title('掩码对比损失变化 (改进方案1)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'save_pkl', '掩码损失曲线.png'), dpi=300)
    plt.close()

print("所有图表已生成！保存在 save_pkl 文件夹中")
