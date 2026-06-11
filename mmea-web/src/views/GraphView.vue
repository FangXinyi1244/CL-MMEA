<template>
  <div class="graph-view">
    <!-- 状态提示 -->
    <div v-if="connectionStatus === 'checking'" class="status checking">
      <span class="spinner"></span> 连接中...
    </div>
    <div v-else-if="connectionStatus === 'error'" class="status error">
      连接失败: {{ errorMessage }}
      <button @click="checkConnection">重试</button>
    </div>

    <div class="layout">
      <!-- 侧边栏 -->
      <aside class="sidebar">
        <h2>🔍 知识图谱</h2>
        
        <div class="search-box">
          <input 
            v-model="searchKeyword" 
            placeholder="搜索实体..." 
            @keyup.enter="handleSearch"
            :disabled="!connected"
          />
          <button @click="handleSearch" :disabled="!connected">搜索</button>
        </div>

        <div class="stats" v-if="stats.length">
          <h3>统计</h3>
          <div v-for="s in stats" :key="s.label" class="stat-item">
            <span>{{ s.label }}</span>
            <strong>{{ s.count }}</strong>
          </div>
        </div>

        <div class="node-info" v-if="selectedNode">
          <h3>实体详情</h3>
          <p><strong>ID:</strong> {{ selectedNode.entityId }}</p>
          <p><strong>名称:</strong> {{ selectedNode.name }}</p>
          <p><strong>数据集:</strong> {{ selectedNode.dataset }}</p>
          <p v-if="selectedNode.neighbors?.length">
            <strong>关联:</strong> {{ selectedNode.neighbors.length }}个
          </p>
          <button class="close-btn" @click="selectedNode = null">关闭</button>
        </div>
      </aside>

      <!-- 图谱 -->
      <main class="graph-area">
        <div class="controls">
          <button @click="resetView">🔄 重置</button>
          <span class="count">节点: {{ graphData.nodes.length }}</span>
        </div>
        <div ref="graphRef" class="canvas"></div>
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import ForceGraph from 'force-graph'
import { checkHealth, getGraphStats, getFullGraph, searchEntity, getEntityDetail } from '../api/index.js'

const connectionStatus = ref('checking')
const errorMessage = ref('')
const connected = ref(false)
const searchKeyword = ref('')
const graphData = ref({ nodes: [], links: [] })
const selectedNode = ref(null)
const stats = ref([])
const graphRef = ref(null)
let graphInstance = null

const colors = { DBP15K_ZH: '#4CAF50', DBP15K_EN: '#2196F3' }
const getColor = (n) => colors[n.dataset] || '#888'

const checkConnection = async () => {
  connectionStatus.value = 'checking'
  try {
    await checkHealth()
    connected.value = true
    connectionStatus.value = 'connected'
    await loadData()
  } catch (e) {
    connectionStatus.value = 'error'
    errorMessage.value = e.message
  }
}

const loadData = async () => {
  const [s, g] = await Promise.all([getGraphStats(), getFullGraph(100)])
  stats.value = s.data || []
  graphData.value = g.data || { nodes: [], links: [] }
  nextTick(() => graphInstance?.graphData(graphData.value))
}

const handleSearch = async () => {
  if (!searchKeyword.value) {
    loadData()
    return
  }
  const res = await searchEntity(searchKeyword.value)
  graphData.value = res.data || { nodes: [], links: [] }
  graphInstance?.graphData(graphData.value)
}

const resetView = () => graphInstance?.zoomToFit(400, 20)

const initGraph = () => {
  if (!graphRef.value) return
  graphInstance = ForceGraph()(graphRef.value)
    .nodeId('id')
    .nodeLabel(n => n.name)
    .nodeColor(getColor)
    .nodeVal(2)
    .linkDirectionalArrowLength(6)
    .onNodeClick(async (n) => {
      const res = await getEntityDetail(n.id)
      selectedNode.value = res.data
    })
}

onMounted(() => { initGraph(); checkConnection() })
onUnmounted(() => graphInstance?._destructor())
</script>

<style scoped>
.graph-view { height: calc(100vh - 60px); display: flex; flex-direction: column; }
.status { padding: 10px; text-align: center; }
.status.checking { background: #fff3e0; }
.status.error { background: #ffebee; color: #c62828; }
.status.error button { margin-left: 10px; padding: 4px 12px; }
.spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid currentColor; border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.layout { display: flex; flex: 1; overflow: hidden; }
.sidebar { width: 300px; background: #f8f9fa; padding: 20px; overflow-y: auto; border-right: 1px solid #ddd; }
.sidebar h2 { margin-bottom: 15px; font-size: 1.1rem; }
.sidebar h3 { margin: 15px 0 10px; font-size: 0.95rem; color: #555; }

.search-box { display: flex; gap: 8px; margin-bottom: 15px; }
.search-box input { flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
.search-box button { padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; }
.search-box button:disabled { background: #aaa; }

.stats { background: white; padding: 12px; border-radius: 6px; margin-bottom: 15px; }
.stat-item { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee; }
.stat-item:last-child { border-bottom: none; }

.node-info { background: white; padding: 15px; border-radius: 6px; }
.node-info p { margin-bottom: 8px; font-size: 0.9rem; }
.close-btn { width: 100%; padding: 8px; margin-top: 10px; background: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer; }

.graph-area { flex: 1; position: relative; background: white; }
.controls { position: absolute; top: 15px; left: 15px; z-index: 10; display: flex; gap: 10px; align-items: center; }
.controls button { padding: 6px 12px; background: white; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }
.count { background: white; padding: 6px 12px; border-radius: 4px; border: 1px solid #ddd; font-size: 0.9rem; }
.canvas { width: 100%; height: 100%; }
</style>
