# CL-MMEA

**Multi-modal Contrastive Representation Learning for Entity Alignment (MCLEA)**

基于多模态对比学习的中英知识图谱实体对齐系统。该模型在 COLING 2022（oral）上发表 [[arxiv](https://arxiv.org/abs/2209.00891)] [[acl](https://aclanthology.org/2022.coling-1.227.pdf)]。

本项目在原论文基础上增加了 **Flask REST API 后端**、**Vue3 可视化前端**、**Neo4j 图数据库集成**、**特征掩码（Feature Masking）** 和 **硬负采样（Hard Negative Sampling）** 改进方案。

---

## 项目结构

```
CL-MMEA/
├── main.py                   # Flask 后端 API 入口（懒加载模式）
├── demo.py                   # 命令行推理演示
├── neo4j_import.py           # Neo4j 图数据库导入工具
├── run_dbp15k.sh             # DBP15K 双语数据集训练脚本
├── run_mmkb.sh               # MMKB 跨知识图谱数据集训练脚本
├── requirements.txt          # Python 依赖
├── src/                      # 核心源码
│   ├── run.py                # MCLEA 类 — 训练/推理主逻辑
│   ├── models.py             # 多模态编码器、融合层、特征掩码
│   ├── layers.py             # GAT / GCN 图神经网络层
│   ├── loss.py               # ICL 对比损失、IAL 对齐损失、掩码对比损失
│   ├── Load.py               # 数据加载器（实体、关系、属性、图像、词向量等）
│   └── utils.py              # 工具函数（邻接矩阵、距离计算、CSLS、日志）
├── datapro/
│   └── mmkb_convert.py       # MMKB 数据集格式转换为 DBP15K 格式
├── data/                     # 数据集目录（需自行下载）
│   ├── DBP15K/               # DBP15K 双语数据集 (zh_en, ja_en, fr_en)
│   ├── mmkb-datasets/        # 跨知识图谱数据集 (FB15K_DB15K, FB15K_YAGO15K)
│   ├── pkls/                 # DBP15K 图像特征 pickle 文件
│   └── embedding/            # GloVe 词向量文件
├── pkl/                      # 训练好的模型文件保存目录
└── mmea-web/                 # Vue3 前端 Web UI
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.vue
        ├── router/index.js
        ├── api/index.js
        └── views/
            ├── Home.vue          # 首页
            ├── ModelDemo.vue     # 模型演示（实体对齐推理）
            ├── GraphView.vue     # 知识图谱可视化（力导向图）
            └── ResultsView.vue   # 实验结果对比展示
```

---

## 模型架构

### 多模态特征

模型同时利用 6 种模态信息进行实体表示：

| 模态 | 说明 | 编码方式 |
|------|------|----------|
| **结构 (Structure)** | 知识图谱拓扑结构 | 3 层 GAT（多头图注意力）或 GCN |
| **图像 (Image)** | 实体关联图片 | 预训练 VGG16 提取特征 |
| **关系 (Relation)** | 实体关联的关系类型统计 | Top-1000 关系分布 → 100 维投影 |
| **属性 (Attribute)** | 实体属性 URI 统计 | Top-1000 属性分布 → 100 维投影 |
| **名称 (Name)** | 翻译后的实体名称 | GloVe 词向量平均 → 100 维投影 |
| **字符 (Character)** | 实体名称字符级特征 | 字符 Bigram → 100 维投影 |

### 核心模块

- **多模态融合层** (`MultiModalFusion`)：使用可学习 softmax 权重对 6 种模态嵌入加权拼接，得到联合嵌入（600 维）
- **三层损失联合优化**：
  - **ICL**（跨模态对比损失）：拉近对齐实体对在各模态空间中的距离
  - **IAL**（模态间对齐损失）：KL 散度对齐单模态与联合嵌入
  - **掩码对比损失**：自监督，对特征随机掩码后保持实体一致性

### 改进方案

- **特征掩码**：训练时以 15% 比例随机掩码部分模态特征维度，提升缺失模态鲁棒性
- **硬负采样**：ICL 损失中对 Top-50 相似负样本施加更高权重（1.5x），增强区分能力

---

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.7.0
- Neo4j 4.x（可选，用于图谱可视化）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据集准备

#### DBP15K 数据集

1. 下载 DBP15K 多模态数据：参照 [EVA](https://github.com/cambridgeltl/eva) 仓库指引
2. 将图像特征 `pkls` 文件夹放入 `data/` 目录
3. 下载 GloVe 词向量并解压到 `data/embedding/`：
   ```bash
   # 从 https://nlp.stanford.edu/data/glove.6B.zip 下载
   unzip glove.6B.zip -d data/embedding/
   ```

#### MMKB 跨知识图谱数据集

1. 下载原始数据：参照 [MMKB](https://github.com/mniepert/mmkb) 仓库
2. 使用百度网盘下载转换后的数据（密码 `stdt`），放入 `data/` 目录
3. 或自行转换：
   ```bash
   python datapro/mmkb_convert.py
   ```

---

## 训练

### DBP15K 双语数据集

```bash
# 参数: GPU_ID 随机种子 语言对
bash run_dbp15k.sh 0 42 zh_en   # 中文-英文
bash run_dbp15k.sh 0 42 ja_en   # 日文-英文
bash run_dbp15k.sh 0 42 fr_en   # 法文-英文
```

### MMKB 跨知识图谱数据集

```bash
# 参数: GPU_ID 随机种子 数据集名 训练集比例(0.2/0.5/0.8)
bash run_mmkb.sh 0 42 FB15K_DB15K 0.2
bash run_mmkb.sh 0 42 FB15K_DB15K 0.5
bash run_mmkb.sh 0 42 FB15K_DB15K 0.8

bash run_mmkb.sh 0 42 FB15K_YAGO15K 0.2
bash run_mmkb.sh 0 42 FB15K_YAGO15K 0.5
bash run_mmkb.sh 0 42 FB15K_YAGO15K 0.8
```

### 关键训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1000 | 训练轮次 |
| `--lr` | 0.0005 | 学习率 |
| `--bsize` | 512 | 批次大小 |
| `--hidden_units` | "300,300,300" | GAT/GCN 隐藏层维度 |
| `--structure_encoder` | "gat" | 图编码器类型（gat / gcn） |
| `--rate` | 0.3 | 训练集比例 |
| `--tau` | 0.1 | ICL 温度系数 |
| `--tau2` | 4.0 | IAL 温度系数 |
| `--il` / `--il_start` | True / 500 | 迭代学习（半监督自训练） |
| `--csls` / `--csls_k` | True / 3 | 推理时 CSLS 度量 |
| `--mask_ratio` | 0.15 | 特征掩码比例 |
| `--use_hard_negatives` | True | 启用硬负采样 |
| `--hard_negative_k` | 50 | 硬负样本 Top-K |
| `--mask_loss_weight` | 0.1 | 掩码对比损失权重 |

---

## 推理

### 命令行演示

```bash
python demo.py --model_path pkl/model_epoch_999_zh_en.pkl --file_dir data/DBP15K/zh_en
```

### Web 服务部署

#### 1. 启动 Neo4j（可选，用于图谱可视化）

确保 Neo4j 图数据库已运行，然后导入数据：

```bash
python neo4j_import.py --dataset dbp15k --language zh_en --clear
```

#### 2. 启动 Flask 后端 API

```bash
python main.py --port 5000
```

环境变量配置：
- `NEO4J_URI`：Neo4j 连接地址（默认 `bolt://localhost:7687`）
- `NEO4J_USER`：Neo4j 用户名（默认 `neo4j`）
- `NEO4J_PASSWORD`：Neo4j 密码（默认 `password`）

使用 `--no-neo4j` 可跳过 Neo4j 连接。

#### 3. 启动 Vue3 前端

```bash
cd mmea-web
npm install
npm run dev
```

访问 `http://localhost:5173` 即可使用 Web UI。

---

## Web 功能

| 功能页面 | 说明 |
|----------|------|
| **知识图谱** | 3D 力导向图可视化，支持搜索实体、查看节点详情、图谱统计 |
| **模型演示** | 懒加载模型、实体对齐推理、相似度分数、多模态注意力权重可视化 |
| **实验结果** | 三种方案（基线 / 特征掩码 / 掩码+硬负采样）对比图表 |

### API 接口一览

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/health` | 健康检查（模型状态 + Neo4j 状态） |
| GET | `/api/models` | 获取可用模型列表 |
| GET | `/api/models/<id>/info` | 获取模型详情 |
| POST | `/api/models/<id>/switch` | 切换/懒加载模型 |
| GET | `/api/model/info` | 当前已加载模型信息 |
| POST | `/api/model/align` | 单对实体对齐推理 |
| POST | `/api/model/batch-align` | 批量实体对齐 |
| POST | `/api/model/topk` | Top-K 相似实体搜索 |
| GET | `/api/model/evaluate` | 模型性能评估（Hits@1/10/50, MRR） |
| GET | `/api/graph/stats` | Neo4j 图谱统计 |
| GET | `/api/graph/full` | 获取完整图谱数据 |
| GET | `/api/graph/search` | 搜索实体 |
| GET | `/api/graph/entity/<id>` | 实体详情及邻居 |
| GET | `/api/graph/alignment` | 获取对齐关系 |

---

## 致谢

本项目代码基于 [MCLEA](https://github.com/lzxlin/MCLEA) 修改，感谢其开源工作。
