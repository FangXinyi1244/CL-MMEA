"""
MMEA 多模态实体对齐系统 - 后端API服务 (懒加载模式)
基于Flask提供模型推理和Neo4j数据库接口

启动方式:
    python main.py
    
环境变量:
    NEO4J_URI: Neo4j连接URI (默认: bolt://localhost:7687)
    NEO4J_USER: Neo4j用户名 (默认: neo4j)
    NEO4J_PASSWORD: Neo4j密码 (默认: password)
    CUDA: 是否使用GPU (默认: false)
"""

import os
import sys
import time
import json
import argparse
import gc
from functools import wraps

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase

# ========== Flask应用配置 ==========
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ========== 全局配置 ==========
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')
USE_CUDA = os.environ.get('CUDA', 'false').lower() == 'true'

# ========== 模型配置 ==========
AVAILABLE_MODELS = {
    "zh_en": {
        "name": "DBP15K 中文-英文",
        "model_file": "model_epoch_999_zh_en.pkl",
        "data_dir": "data/DBP15K/zh_en",
        "description": "DBPedia中文到英文实体对齐模型",
        "left_range": [0, 10499],
        "right_range": [10500, 20999]
    },
    "FB15K_DB15K": {
        "name": "FB15K-DB15K",
        "model_file": "model_epoch_999_FB15K_DB15K.pkl", 
        "data_dir": "data/mmkb-datasets/FB15K_DB15K",
        "description": "FreeBase到DBPedia实体对齐模型",
        "left_range": [0, 14999],
        "right_range": [15000, 29999]
    },
    "FB15K_YAGO15K": {
        "name": "FB15K-YAGO15K",
        "model_file": "model_epoch_999_FB15K_YAGO15K.pkl",
        "data_dir": "data/mmkb-datasets/FB15K_YAGO15K", 
        "description": "FreeBase到YAGO实体对齐模型",
        "left_range": [0, 14999],
        "right_range": [15000, 29999]
    }
}

# ========== 全局状态 ==========
neo4j_driver = None
current_model = None      # 当前模型实例
current_model_id = None   # 当前模型ID
current_embeddings = None # 当前模型嵌入


def get_pkl_dir():
    """获取pkl目录路径"""
    return os.path.join(os.path.dirname(__file__), "pkl")


def discover_models():
    """发现可用的模型文件"""
    pkl_dir = get_pkl_dir()
    available = {}
    
    for model_id, config in AVAILABLE_MODELS.items():
        model_path = os.path.join(pkl_dir, config["model_file"])
        config_copy = config.copy()
        config_copy["exists"] = os.path.exists(model_path)
        config_copy["path"] = model_path if os.path.exists(model_path) else None
        available[model_id] = config_copy
    
    return available


def release_model():
    """释放当前模型占用的资源"""
    global current_model, current_embeddings, current_model_id
    
    if current_model is not None:
        print(f"\n释放模型资源: {current_model_id}")
        # 删除模型引用
        del current_model
        del current_embeddings
        current_model = None
        current_embeddings = None
        current_model_id = None
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("资源释放完成")


def load_model_lazy(model_id, cuda=False):
    """
    懒加载指定模型
    Args:
        model_id: 模型ID
        cuda: 是否使用GPU (默认CPU避免OOM)
    Returns:
        (success, error_msg)
    """
    global current_model, current_embeddings, current_model_id
    
    # 如果已加载相同模型，直接返回
    if current_model_id == model_id and current_model is not None:
        return True, None
    
    # 释放已有模型
    if current_model is not None:
        release_model()
    
    # 检查模型是否可用
    discovered = discover_models()
    if model_id not in discovered:
        return False, f"未知模型: {model_id}"
    
    config = discovered[model_id]
    if not config.get("exists"):
        return False, f"模型文件不存在: {config['model_file']}"
    
    model_path = config["path"]
    file_dir = config["data_dir"]
    
    print(f"\n{'='*60}")
    print(f"正在加载模型: {config['name']}")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"数据路径: {file_dir}")
    
    # 强制使用CPU避免CUDA OOM
    device = torch.device("cpu")
    print(f"使用设备: CPU (强制使用CPU避免内存不足)")
    
    try:
        # 延迟导入MCLEA，避免启动时加载
        from run import MCLEA
        
        # 加载保存的字典
        save_dict = torch.load(model_path, map_location=device)
        saved_args = save_dict['args']
        
        # 覆盖参数 - 强制CPU模式
        saved_args['file_dir'] = file_dir
        saved_args['cuda'] = False
        
        # 构建命令行参数
        import sys as sys_module
        original_argv = sys_module.argv[:]
        
        store_false_flags = {'w_gcn', 'w_rel', 'w_attr', 'w_name', 'w_char', 'w_img'}
        
        sys_module.argv = ['main.py']
        for key, value in saved_args.items():
            if isinstance(value, bool):
                if key in store_false_flags:
                    if not value:
                        sys_module.argv.append(f'--{key}')
                else:
                    if value:
                        sys_module.argv.append(f'--{key}')
            else:
                sys_module.argv.append(f'--{key}={value}')
        
        # 创建MCLEA实例
        current_model = MCLEA()
        sys_module.argv = original_argv
        
        # 加载模型参数
        current_model.multimodal_encoder.load_state_dict(save_dict['multimodal_encoder'])
        current_model.multi_loss_layer.load_state_dict(save_dict['multi_loss_layer'])
        current_model.align_multi_loss_layer.load_state_dict(save_dict['align_multi_loss_layer'])
        
        # 确保input_idx初始化
        if current_model.input_idx is None:
            current_model.input_idx = torch.LongTensor(np.arange(current_model.ENT_NUM)).to(current_model.device)
        
        # 设置为评估模式
        current_model.multimodal_encoder.eval()
        
        print("✓ 模型结构加载成功")
        print(f"  - 实体数量: {current_model.ENT_NUM}")
        print(f"  - 关系数量: {current_model.REL_NUM}")
        print(f"  - 训练样本: {current_model.train_ill.shape[0]}")
        print(f"  - 测试样本: {current_model.test_ill.shape[0]}")
        
        # 预计算所有实体的嵌入向量
        print("\n预计算实体嵌入向量...")
        with torch.no_grad():
            *embs, _ = current_model.multimodal_encoder(
                current_model.input_idx,
                current_model.adj,
                current_model.img_features,
                current_model.rel_features,
                current_model.att_features,
                current_model.name_features,
                current_model.char_features
            )
            gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = embs[:7]
            current_embeddings = torch.nn.functional.normalize(joint_emb)
        
        print(f"✓ 嵌入向量计算完成: {current_embeddings.shape}")
        print(f"{'='*60}\n")
        
        current_model_id = model_id
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ 模型加载失败: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # 清理残留
        if current_model is not None:
            release_model()
        return False, error_msg


def format_node(node):
    """格式化Neo4j节点为JSON"""
    if node is None:
        return None
    return {
        "id": str(node.element_id) if hasattr(node, 'element_id') else str(node.get('entityId')),
        "entityId": node.get("entityId"),
        "name": node.get("name"),
        "dataset": node.get("dataset"),
        "uri": node.get("uri"),
        "labels": list(node.labels) if hasattr(node, 'labels') else [],
        "propertyList": node.get("propertyList", [])
    }


# ========== 初始化函数 ==========
def init_neo4j():
    """初始化Neo4j连接"""
    global neo4j_driver
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                print(f"✓ Neo4j连接成功: {NEO4J_URI}")
                return True
    except Exception as e:
        print(f"⚠ Neo4j连接失败: {e}")
        print(f"  URI: {NEO4J_URI}")
        print(f"  部分图谱功能将不可用")
        neo4j_driver = None
        return False


# ========== API路由 ==========

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    status = {
        "status": "ok",
        "model_loaded": current_model is not None,
        "current_model": current_model_id,
        "neo4j_connected": neo4j_driver is not None,
        "timestamp": time.time()
    }
    return jsonify({"success": True, "data": status})


@app.route('/api/models', methods=['GET'])
def list_models():
    """获取可用模型列表"""
    discovered = discover_models()
    model_list = []
    
    for model_id, config in discovered.items():
        info = {
            "id": model_id,
            "name": config["name"],
            "description": config.get("description", ""),
            "exists": config.get("exists", False),
            "loaded": model_id == current_model_id,
            "is_current": model_id == current_model_id,
            "left_range": config.get("left_range"),
            "right_range": config.get("right_range")
        }
        if info["loaded"] and current_model is not None:
            info["entity_num"] = current_model.ENT_NUM
            info["relation_num"] = current_model.REL_NUM
        model_list.append(info)
    
    return jsonify({"success": True, "data": model_list})


@app.route('/api/models/<model_id>/switch', methods=['POST'])
def switch_model_api(model_id):
    """切换到指定模型（懒加载）"""
    if model_id not in AVAILABLE_MODELS:
        return jsonify({"success": False, "error": f"未知模型ID: {model_id}"}), 400
    
    # 如果已经加载，直接返回
    if current_model_id == model_id and current_model is not None:
        return jsonify({"success": True, "data": {"current_model": model_id, "note": "模型已加载"}})
    
    # 加载新模型
    success, error = load_model_lazy(model_id, cuda=False)
    if success:
        return jsonify({"success": True, "data": {"current_model": model_id}})
    else:
        return jsonify({"success": False, "error": error}), 500


@app.route('/api/models/<model_id>/info', methods=['GET'])
def get_specific_model_info(model_id):
    """获取指定模型的详细信息"""
    if model_id not in AVAILABLE_MODELS:
        return jsonify({"success": False, "error": "未知模型ID"}), 400
    
    discovered = discover_models()
    if model_id not in discovered:
        return jsonify({"success": False, "error": "模型配置错误"}), 500
    
    config = discovered[model_id]
    
    info = {
        "id": model_id,
        "name": config["name"],
        "description": config.get("description", ""),
        "exists": config.get("exists", False),
        "loaded": model_id == current_model_id,
        "left_range": config.get("left_range"),
        "right_range": config.get("right_range")
    }
    
    if current_model_id == model_id and current_model is not None:
        info["entity_num"] = current_model.ENT_NUM
        info["relation_num"] = current_model.REL_NUM
    
    return jsonify({"success": True, "data": info})


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """获取当前模型信息"""
    if current_model is None:
        return jsonify({"success": True, "data": {"status": "未加载", "loaded": False}})
    
    config = AVAILABLE_MODELS.get(current_model_id, {})
    info = {
        "model_id": current_model_id,
        "model_name": "MCLEA",
        "version": "2.1",
        "name": config.get("name", "Unknown"),
        "loaded": True,
        "entity_num": current_model.ENT_NUM,
        "relation_num": current_model.REL_NUM,
        "left_range": config.get("left_range"),
        "right_range": config.get("right_range")
    }
    
    return jsonify({"success": True, "data": info})


# ========== 知识图谱接口 ==========

@app.route('/api/graph/stats', methods=['GET'])
def graph_stats():
    """获取图谱统计信息"""
    if neo4j_driver is None:
        return jsonify({"success": False, "error": "Neo4j未连接"}), 503
    
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (e:Entity) 
                RETURN e.dataset as dataset, count(e) as cnt
                ORDER BY dataset
            """)
            stats = [{"label": r["dataset"], "count": r["cnt"]} for r in result]
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/graph/full', methods=['GET'])
def graph_full():
    """获取完整图谱"""
    if neo4j_driver is None:
        return jsonify({"success": False, "error": "Neo4j未连接"}), 503
    
    limit = request.args.get('limit', 100, type=int)
    
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (n:Entity)-[r]->(m:Entity)
                RETURN n, r, m 
                LIMIT $limit
            """, limit=limit)
            
            nodes = {}
            links = []
            for record in result:
                n, m, rel = record["n"], record["m"], record["r"]
                n_id = str(n.get("entityId"))
                m_id = str(m.get("entityId"))
                
                if n_id not in nodes:
                    nodes[n_id] = format_node(n)
                if m_id not in nodes:
                    nodes[m_id] = format_node(m)
                    
                links.append({
                    "source": n_id,
                    "target": m_id,
                    "type": rel.type if hasattr(rel, 'type') else "RELATED"
                })
        
        return jsonify({
            "success": True, 
            "data": {
                "nodes": list(nodes.values()), 
                "links": links
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/graph/search', methods=['GET'])
def graph_search():
    """搜索实体"""
    if neo4j_driver is None:
        return jsonify({"success": False, "error": "Neo4j未连接"}), 503
    
    keyword = request.args.get('keyword', '')
    if not keyword:
        return jsonify({"success": False, "error": "缺少keyword参数"}), 400
    
    try:
        with neo4j_driver.session() as session:
            try:
                entity_id = int(keyword)
                id_search = True
            except ValueError:
                entity_id = 0
                id_search = False
            
            result = session.run("""
                MATCH (n:Entity)
                WHERE ($id_search = true AND n.entityId = $entity_id)
                   OR n.name CONTAINS $keyword
                   OR n.uri CONTAINS $keyword
                OPTIONAL MATCH (n)-[r]-(m:Entity)
                RETURN n, r, m 
                LIMIT 50
            """, keyword=keyword, entity_id=entity_id, id_search=id_search)
            
            nodes = {}
            links = []
            for record in result:
                n = record["n"]
                m = record["m"]
                rel = record["r"]
                
                n_id = str(n.get("entityId"))
                if n_id not in nodes:
                    nodes[n_id] = format_node(n)
                
                if m is not None and rel is not None:
                    m_id = str(m.get("entityId"))
                    if m_id not in nodes:
                        nodes[m_id] = format_node(m)
                    links.append({
                        "source": n_id,
                        "target": m_id,
                        "type": rel.type if hasattr(rel, 'type') else "RELATED"
                    })
        
        return jsonify({
            "success": True, 
            "data": {
                "nodes": list(nodes.values()), 
                "links": links
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/graph/entity/<entity_id>', methods=['GET'])
def graph_entity(entity_id):
    """获取实体详情"""
    if neo4j_driver is None:
        return jsonify({"success": False, "error": "Neo4j未连接"}), 503
    
    try:
        entity_id_int = int(entity_id)
    except ValueError:
        return jsonify({"success": False, "error": "无效的entity_id"}), 400
    
    try:
        with neo4j_driver.session() as session:
            node_result = session.run("""
                MATCH (n:Entity {entityId: $id}) 
                RETURN n
            """, id=entity_id_int).single()
            
            if node_result is None:
                return jsonify({"success": False, "error": "实体不存在"}), 404
            
            node = node_result["n"]
            data = format_node(node)
            
            neighbors_result = session.run("""
                MATCH (n:Entity {entityId: $id})-[r]-(m:Entity)
                RETURN type(r) as rel_type, m
                LIMIT 50
            """, id=entity_id_int)
            
            data["neighbors"] = [
                {
                    "relation": n["rel_type"], 
                    "target": format_node(n["m"])
                }
                for n in neighbors_result
            ]
        
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/graph/alignment', methods=['GET'])
def graph_alignment():
    """获取对齐关系"""
    if neo4j_driver is None:
        return jsonify({"success": False, "error": "Neo4j未连接"}), 503
    
    limit = request.args.get('limit', 100, type=int)
    
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (e1:Entity)-[r:ALIGNS_WITH]->(e2:Entity)
                RETURN e1, r, e2
                LIMIT $limit
            """, limit=limit)
            
            alignments = []
            for record in result:
                e1 = record["e1"]
                e2 = record["e2"]
                rel = record["r"]
                alignments.append({
                    "source": format_node(e1),
                    "target": format_node(e2),
                    "confidence": rel.get("confidence", 1.0) if hasattr(rel, 'get') else 1.0
                })
        
        return jsonify({"success": True, "data": alignments})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ========== 模型接口 ==========

def validate_entity_ids(source_id, target_id, model):
    """验证实体ID是否有效"""
    try:
        source_id = int(source_id)
        target_id = int(target_id)
    except ValueError:
        return None, None, "实体ID必须是整数"
    
    if source_id < 0 or source_id >= model.ENT_NUM or target_id < 0 or target_id >= model.ENT_NUM:
        return None, None, f"实体ID超出范围 (0-{model.ENT_NUM-1})"
    
    return source_id, target_id, None


@app.route('/api/model/align', methods=['POST'])
def model_align():
    """执行实体对齐推理"""
    if current_model is None or current_embeddings is None:
        return jsonify({"success": False, "error": "模型未加载，请先选择模型"}), 503
    
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400
    
    source = data.get('source', {})
    target = data.get('target', {})
    
    source_id = source.get('id')
    target_id = target.get('id')
    
    source_id, target_id, error = validate_entity_ids(source_id, target_id, current_model)
    if error:
        return jsonify({"success": False, "error": error}), 400
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            source_emb = current_embeddings[source_id]
            target_emb = current_embeddings[target_id]
            
            distance = torch.norm(source_emb - target_emb, p=2).item()
            similarity = np.exp(-distance)
            
            w_normalized = torch.nn.functional.softmax(current_model.multimodal_encoder.fusion.weight, dim=0)
            weights = w_normalized.data.squeeze().cpu().numpy()
            
            feature_scores = {
                "structural": round(float(weights[0]) * similarity + 0.1, 2),
                "relational": round(float(weights[1]) * similarity + 0.1, 2),
                "attribute": round(float(weights[2]) * similarity + 0.1, 2),
                "visual": round(float(weights[3]) * similarity + 0.1, 2),
                "name": round(float(weights[4]) * similarity + 0.1, 2),
                "character": round(float(weights[5]) * similarity + 0.1, 2)
            }
            
            inference_time = (time.time() - start_time) * 1000
            
            threshold = 0.65
            
            return jsonify({
                "success": True,
                "data": {
                    "similarity": float(round(similarity, 4)),
                    "distance": float(round(distance, 4)),
                    "is_aligned": bool(similarity > threshold),
                    "threshold": float(threshold),
                    "feature_scores": feature_scores,
                    "attention_map": [float(round(w, 3)) for w in weights],
                    "inference_time": float(round(inference_time, 2)),
                    "model_version": f"MCLEA-{current_model_id}",
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "model_id": current_model_id
                }
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/model/batch-align', methods=['POST'])
def model_batch_align():
    """批量实体对齐"""
    if current_model is None or current_embeddings is None:
        return jsonify({"success": False, "error": "模型未加载，请先选择模型"}), 503
    
    data = request.json
    if not data or 'pairs' not in data:
        return jsonify({"success": False, "error": "缺少pairs字段"}), 400
    
    pairs = data['pairs']
    if not isinstance(pairs, list):
        return jsonify({"success": False, "error": "pairs必须是数组"}), 400
    
    start_time = time.time()
    results = []
    threshold = 0.65
    
    try:
        with torch.no_grad():
            for pair in pairs:
                source_id = int(pair.get('source_id', -1))
                target_id = int(pair.get('target_id', -1))
                
                if source_id < 0 or source_id >= current_model.ENT_NUM or target_id < 0 or target_id >= current_model.ENT_NUM:
                    results.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "error": "实体ID超出范围"
                    })
                    continue
                
                source_emb = current_embeddings[source_id]
                target_emb = current_embeddings[target_id]
                distance = torch.norm(source_emb - target_emb, p=2).item()
                similarity = np.exp(-distance)
                
                results.append({
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "similarity": float(round(similarity, 4)),
                    "is_aligned": bool(similarity > threshold)
                })
        
        inference_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "success": True,
            "data": {
                "results": results,
                "total": len(pairs),
                "aligned_count": sum(1 for r in results if r.get('is_aligned', False)),
                "inference_time": round(inference_time, 2),
                "model_id": current_model_id
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/model/topk', methods=['POST'])
def model_topk():
    """为给定实体寻找最相似的Top-K实体"""
    if current_model is None or current_embeddings is None:
        return jsonify({"success": False, "error": "模型未加载，请先选择模型"}), 503
    
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400
    
    entity_id = data.get('entity_id')
    k = data.get('k', 10)
    direction = data.get('direction', 'right')
    
    if entity_id is None:
        return jsonify({"success": False, "error": "缺少entity_id"}), 400
    
    try:
        entity_id = int(entity_id)
        k = int(k)
    except ValueError:
        return jsonify({"success": False, "error": "参数类型错误"}), 400
    
    if entity_id < 0 or entity_id >= current_model.ENT_NUM:
        return jsonify({"success": False, "error": "实体ID超出范围"}), 400
    
    try:
        start_time = time.time()
        
        with torch.no_grad():
            query_emb = current_embeddings[entity_id]
            
            if direction == 'right':
                candidate_ids = current_model.right_ents
            else:
                candidate_ids = current_model.left_ents
            
            candidate_embs = current_embeddings[candidate_ids]
            distances = torch.norm(candidate_embs - query_emb.unsqueeze(0), p=2, dim=1)
            
            topk_values, topk_indices = torch.topk(distances, k=min(k, len(distances)), largest=False)
            
            results = []
            for i, (dist, idx) in enumerate(zip(topk_values, topk_indices)):
                target_id = candidate_ids[idx].item()
                similarity = np.exp(-dist.item())
                results.append({
                    "rank": int(i + 1),
                    "entity_id": int(target_id),
                    "similarity": float(round(similarity, 4)),
                    "is_aligned": bool(similarity > 0.65)
                })
            
            inference_time = (time.time() - start_time) * 1000
            
            return jsonify({
                "success": True,
                "data": {
                    "query_id": entity_id,
                    "direction": direction,
                    "topk": results,
                    "inference_time": round(inference_time, 2),
                    "model_id": current_model_id
                }
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/model/evaluate', methods=['GET'])
def model_evaluate():
    """评估当前模型性能指标"""
    if current_model is None or current_embeddings is None:
        return jsonify({"success": False, "error": "模型未加载，请先选择模型"}), 503
    
    try:
        from utils import pairwise_distances
        
        start_time = time.time()
        
        with torch.no_grad():
            test_left = current_model.test_left
            test_right = current_model.test_right
            
            distance = pairwise_distances(current_embeddings[test_left], current_embeddings[test_right])
            
            top_k = [1, 10, 50]
            acc_l2r = np.zeros(len(top_k), dtype=np.float32)
            mrr_l2r = 0.0
            
            for idx in range(test_left.shape[0]):
                values, indices = torch.sort(distance[idx, :], descending=False)
                rank = (indices == idx).nonzero().squeeze().item()
                mrr_l2r += 1.0 / (rank + 1)
                for i, k in enumerate(top_k):
                    if rank < k:
                        acc_l2r[i] += 1
            
            acc_l2r = acc_l2r / test_left.shape[0]
            mrr_l2r = mrr_l2r / test_left.shape[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            return jsonify({
                "success": True,
                "data": {
                    "hits@1": float(round(acc_l2r[0], 4)),
                    "hits@10": float(round(acc_l2r[1], 4)),
                    "hits@50": float(round(acc_l2r[2], 4)),
                    "mrr": float(round(mrr_l2r, 4)),
                    "test_samples": int(test_left.shape[0]),
                    "inference_time": float(round(inference_time, 2)),
                    "model_id": current_model_id
                }
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ========== 主函数 ==========

def parse_args():
    parser = argparse.ArgumentParser(description="MMEA后端API服务 (懒加载模式)")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务端口')
    parser.add_argument('--no-neo4j', action='store_true', help='不连接Neo4j')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("MMEA 多模态实体对齐系统 - 后端API服务")
    print("模式: 懒加载 (按需加载模型)")
    print("="*60)
    
    # 初始化Neo4j
    if not args.no_neo4j:
        init_neo4j()
    else:
        print("⚠ 已跳过Neo4j连接 (--no-neo4j)")
    
    # 检查可用模型
    discovered = discover_models()
    available_count = sum(1 for c in discovered.values() if c.get("exists"))
    print(f"\n可用模型: {available_count}/{len(discovered)}")
    for model_id, config in discovered.items():
        status = "✓" if config.get("exists") else "✗"
        print(f"  {status} {config['name']}")
    
    # 启动服务
    print("\n" + "="*60)
    print(f"API服务启动于 http://{args.host}:{args.port}")
    print("="*60)
    print("\n注意: 模型采用懒加载模式，首次使用时会自动加载")
    print("\n可用接口:")
    print("  GET  /api/health              - 健康检查")
    print("  GET  /api/models              - 模型列表")
    print("  GET  /api/models/<id>/info    - 模型详情")
    print("  POST /api/models/<id>/switch  - 加载/切换模型")
    print("  GET  /api/model/info          - 当前模型信息")
    print("  POST /api/model/align         - 实体对齐")
    print("  POST /api/model/batch-align   - 批量对齐")
    print("  POST /api/model/topk          - Top-K相似实体")
    print("  GET  /api/model/evaluate      - 模型评估")
    print("  GET  /api/graph/stats         - 图谱统计")
    print("  GET  /api/graph/full          - 完整图谱")
    print("  GET  /api/graph/search        - 实体搜索")
    print("  GET  /api/graph/entity/<id>   - 实体详情")
    print("  GET  /api/graph/alignment     - 对齐关系")
    print("="*60 + "\n")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
