# MMEA 后端接口规范（简化版）

> 版本: 1.1 (简化版)  
> 日期: 2026-05-28  

---

## 1. 基础信息

### 1.1 服务配置

| 配置项 | 说明 |
|--------|------|
| 建议端口 | 5000 |
| 数据格式 | JSON |
| CORS | 允许 `http://localhost:5173` |

### 1.2 统一响应格式

```json
{
  "success": true,
  "data": {}
}
```

---

## 2. 接口列表（共6个）

### 2.1 健康检查

**GET** `/api/health`

**响应:**
```json
{
  "success": true,
  "data": { "status": "ok" }
}
```

---

### 2.2 知识图谱接口

#### GET `/api/graph/stats`
获取图谱统计信息

**响应:**
```json
{
  "success": true,
  "data": [
    { "label": "DBP15K_ZH", "count": 15000 },
    { "label": "DBP15K_EN", "count": 15000 }
  ]
}
```

---

#### GET `/api/graph/full?limit=100`
获取完整图谱

**参数:**
- `limit`: 限制节点数量 (默认100)

**响应:**
```json
{
  "success": true,
  "data": {
    "nodes": [
      {
        "id": "10001",
        "entityId": 10001,
        "name": "苹果公司",
        "dataset": "DBP15K_ZH",
        "uri": "http://zh.dbpedia.org/苹果公司",
        "labels": ["Entity", "DBP15K_ZH"]
      }
    ],
    "links": [
      {
        "source": "10001",
        "target": "10004",
        "type": "competitor"
      }
    ]
  }
}
```

---

#### GET `/api/graph/search?keyword=xxx`
搜索实体

**响应:** 格式同 `/api/graph/full`

---

#### GET `/api/graph/entity/{entityId}`
获取实体详情

**响应:**
```json
{
  "success": true,
  "data": {
    "id": "10001",
    "entityId": 10001,
    "name": "苹果公司",
    "dataset": "DBP15K_ZH",
    "uri": "...",
    "labels": ["Entity", "DBP15K_ZH"],
    "propertyList": ["公司", "科技企业"],
    "neighbors": [
      {
        "relation": "competitor",
        "target": { "entityId": 10004, "name": "阿里巴巴", "dataset": "DBP15K_ZH" }
      }
    ]
  }
}
```

---

### 2.3 模型接口

#### GET `/api/model/info`
获取模型信息

**响应:**
```json
{
  "success": true,
  "data": {
    "model_name": "MCLEA",
    "version": "2.1",
    "threshold": 0.65
  }
}
```

---

#### POST `/api/model/align`
执行实体对齐

**请求体:**
```json
{
  "source": {
    "id": "10001",
    "name": "苹果公司",
    "text": "苹果公司描述文本"
  },
  "target": {
    "id": "20001",
    "name": "Apple Inc.",
    "text": "Apple Inc. description"
  }
}
```

**响应:**
```json
{
  "success": true,
  "data": {
    "similarity": 0.8923,
    "is_aligned": true,
    "threshold": 0.65,
    "feature_scores": {
      "visual": 0.85,
      "textual": 0.92,
      "structural": 0.78
    },
    "inference_time": 125.5,
    "model_version": "MCLEA-v2.1"
  }
}
```

---

## 3. Python 后端最小实现

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Neo4j 配置
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# ========== 健康检查 ==========
@app.route('/api/health')
def health():
    return jsonify({"success": True, "data": {"status": "ok"}})

# ========== 图谱接口 ==========

def format_node(node):
    return {
        "id": str(node.element_id),
        "entityId": node.get("entityId"),
        "name": node.get("name"),
        "dataset": node.get("dataset"),
        "uri": node.get("uri"),
        "labels": list(node.labels),
        "propertyList": node.get("propertyList", [])
    }

@app.route('/api/graph/stats')
def graph_stats():
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity) 
            RETURN e.dataset as dataset, count(e) as cnt
        """)
        stats = [{"label": r["dataset"], "count": r["cnt"]} for r in result]
    return jsonify({"success": True, "data": stats})

@app.route('/api/graph/full')
def graph_full():
    limit = request.args.get('limit', 100, type=int)
    with driver.session() as session:
        result = session.run("""
            MATCH (n:Entity)-[r]->(m:Entity)
            RETURN n, r, m LIMIT $limit
        """, limit=limit)
        
        nodes = {}
        links = []
        for r in result:
            n, m, rel = r["n"], r["m"], r["r"]
            if n.element_id not in nodes:
                nodes[n.element_id] = format_node(n)
            if m.element_id not in nodes:
                nodes[m.element_id] = format_node(m)
            links.append({
                "source": str(n.get("entityId")),
                "target": str(m.get("entityId")),
                "type": rel.type
            })
    
    return jsonify({"success": True, "data": {"nodes": list(nodes.values()), "links": links}})

@app.route('/api/graph/search')
def graph_search():
    keyword = request.args.get('keyword', '')
    with driver.session() as session:
        result = session.run("""
            MATCH (n:Entity)
            WHERE n.name CONTAINS $keyword OR n.entityId = $id
            OPTIONAL MATCH (n)-[r]-(m:Entity)
            RETURN n, r, m LIMIT 50
        """, keyword=keyword, id=int(keyword) if keyword.isdigit() else 0)
        # ... 格式化同 graph_full
    return jsonify({"success": True, "data": data})

@app.route('/api/graph/entity/<entity_id>')
def graph_entity(entity_id):
    with driver.session() as session:
        # 获取实体
        node = session.run("""
            MATCH (n:Entity {entityId: $id}) RETURN n
        """, id=int(entity_id)).single()["n"]
        
        data = format_node(node)
        
        # 获取邻居
        neighbors = session.run("""
            MATCH (n:Entity {entityId: $id})-[r]-(m:Entity)
            RETURN type(r) as rel, m
        """, id=int(entity_id))
        
        data["neighbors"] = [
            {"relation": n["rel"], "target": format_node(n["m"])}
            for n in neighbors
        ]
    
    return jsonify({"success": True, "data": data})

# ========== 模型接口 ==========
import random

@app.route('/api/model/info')
def model_info():
    return jsonify({
        "success": True,
        "data": {
            "model_name": "MCLEA",
            "version": "2.1",
            "threshold": 0.65
        }
    })

@app.route('/api/model/align', methods=['POST'])
def model_align():
    data = request.json
    # TODO: 替换为真实模型推理
    sim = random.uniform(0.6, 0.95)
    
    return jsonify({
        "success": True,
        "data": {
            "similarity": round(sim, 4),
            "is_aligned": sim > 0.65,
            "threshold": 0.65,
            "feature_scores": {
                "visual": round(random.uniform(0.6, 0.9), 2),
                "textual": round(random.uniform(0.6, 0.9), 2),
                "structural": round(random.uniform(0.5, 0.8), 2)
            },
            "inference_time": round(random.uniform(50, 150), 1),
            "model_version": "MCLEA-v2.1"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## 4. 安装与启动

```bash
# 安装依赖
pip install flask flask-cors neo4j-driver

# 启动服务
python app.py
```

---

## 5. 前端配置

创建 `.env` 文件:

```
VITE_API_URL=http://localhost:5000
```

前端代码已简化，仅调用上述6个接口。
