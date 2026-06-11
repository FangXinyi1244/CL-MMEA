// API 统一出口
// 配置：复制 .env.example 为 .env，设置 VITE_API_URL

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

// 通用请求
async function request(url, options = {}) {
  const response = await fetch(`${API_BASE_URL}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

// ========== 健康检查 ==========
export const checkHealth = () => request('/api/health')

// ========== 模型管理 ==========
export const getModels = () => request('/api/models')
export const getModelInfo = (modelId) => {
  if (modelId) {
    return request(`/api/models/${modelId}/info`)
  }
  return request('/api/model/info')
}
export const switchModel = (modelId) => request(`/api/models/${modelId}/switch`, {
  method: 'POST'
})

// ========== 知识图谱 ==========
export const getGraphStats = () => request('/api/graph/stats')
export const getFullGraph = (limit = 100) => request(`/api/graph/full?limit=${limit}`)
export const searchEntity = (keyword) => request(`/api/graph/search?keyword=${encodeURIComponent(keyword)}`)
export const getEntityDetail = (entityId) => request(`/api/graph/entity/${entityId}`)

// ========== 模型推理 ==========
export const alignEntities = (source, target) => request('/api/model/align', {
  method: 'POST',
  body: JSON.stringify({ source, target })
})

export const batchAlign = (pairs) => request('/api/model/batch-align', {
  method: 'POST',
  body: JSON.stringify({ pairs })
})

export const getTopK = (entityId, k = 10, direction = 'right') => request('/api/model/topk', {
  method: 'POST',
  body: JSON.stringify({ entity_id: entityId, k, direction })
})

export const evaluateModel = () => request('/api/model/evaluate')
