# MMEA 前端项目 - 后端接口文档

## 概述

本文档描述多模态实体对齐演示系统所需的后端接口规范。

## 基础信息

- **服务地址**: `http://localhost:5000` (可配置)
- **数据格式**: JSON
- **字符编码**: UTF-8
- **CORS**: 必须启用，允许前端域名访问

## 接口列表

### 1. 实体对齐接口

**URL**: `POST /api/align`

**功能**: 对两个实体进行多模态对齐，返回相似度分数。

**请求体**:
```json
{
  "source": {
    "id": "10001",
    "name": "苹果公司",
    "image_url": "/path/to/image.jpg",
    "text": "苹果公司是一家美国跨国科技公司..."
  },
  "target": {
    "id": "20001", 
    "name": "Apple Inc.",
    "image_url": "/path/to/image2.jpg",
    "text": "Apple Inc. is an American multinational technology company..."
  }
}
```

**响应体**:
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
      "structural": 0.78,
      "attribute": 0.88
    },
    "attention_map": [0.3, 0.4, 0.3],
    "inference_time": 125.5,
    "model_version": "MCLEA-v2.1"
  }
}
```

### 2. 批量对齐接口

**URL**: `POST /api/batch-align`

**请求体**:
```json
{
  "pairs": [
    {"source_id": "e1", "target_id": "e2"},
    {"source_id": "e3", "target_id": "e4"}
  ]
}
```

### 3. 获取模型信息

**URL**: `GET /api/model-info`

**响应体**:
```json
{
  "success": true,
  "data": {
    "model_name": "MCLEA",
    "version": "2.1",
    "supported_modalities": ["image", "text", "structure"],
    "input_dim": 768,
    "parameters": "15M"
  }
}
```

### 4. 获取实体特征

**URL**: `GET /api/entity-features?entity_id=xxx`

## Python Flask 后端参考实现

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import time

app = Flask(__name__)
CORS(app)  # 允许跨域

# 加载你的预训练模型
# model = load_your_model('checkpoint.pth')

def extract_features(entity_data):
    """
    从实体数据中提取多模态特征
    根据你的模型实现特征提取逻辑
    """
    features = {}
    
    # 视觉特征
    if entity_data.get('image_url'):
        # features['visual'] = image_encoder(entity_data['image_url'])
        pass
    
    # 文本特征  
    if entity_data.get('text'):
        # features['textual'] = text_encoder(entity_data['text'])
        pass
    
    # 结构特征（从Neo4j获取邻居信息）
    # features['structural'] = get_structural_features(entity_data['id'])
    
    return features

@app.route('/api/align', methods=['POST'])
def align():
    data = request.json
    source = data.get('source', {})
    target = data.get('target', {})
    
    # 记录开始时间
    start_time = time.time()
    
    # 1. 提取特征
    source_features = extract_features(source)
    target_features = extract_features(target)
    
    # 2. 模型推理 (这里使用模拟数据，替换为你的实际模型)
    # with torch.no_grad():
    #     similarity, feature_scores, attention = model(source_features, target_features)
    
    # 模拟结果
    similarity = 0.85  # 替换为模型输出
    
    inference_time = (time.time() - start_time) * 1000
    
    return jsonify({
        'success': True,
        'data': {
            'similarity': similarity,
            'is_aligned': similarity > 0.65,
            'threshold': 0.65,
            'feature_scores': {
                'visual': 0.82,
                'textual': 0.88,
                'structural': 0.75,
                'attribute': 0.80
            },
            'attention_map': [0.25, 0.45, 0.30],
            'inference_time': round(inference_time, 2),
            'model_version': 'MCLEA-v2.1'
        }
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'success': True,
        'data': {
            'model_name': 'MCLEA',
            'version': '2.1',
            'supported_modalities': ['image', 'text', 'structure'],
            'input_dim': 768,
            'parameters': '15M'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## 环境配置

1. 安装依赖:
```bash
pip install flask flask-cors torch
```

2. 启动服务:
```bash
python app.py
```

3. 前端配置:
复制 `.env.example` 为 `.env`，修改 `VITE_API_URL` 为你的后端地址。

## 注意事项

1. **CORS配置**: 必须允许前端域名访问后端API
2. **图像处理**: 建议实现图像上传接口，或直接使用base64编码传输
3. **性能优化**: 批量对齐接口建议使用异步队列处理大量请求
4. **错误处理**: 所有接口应返回 `success` 字段标识请求状态
