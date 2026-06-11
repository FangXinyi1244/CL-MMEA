<template>
  <div class="model-demo">
    <h1>⚙️ 模型演示</h1>
    
    <!-- 状态 -->
    <div v-if="apiStatus === 'checking'" class="status checking">检测服务中...</div>
    <div v-else-if="apiStatus === 'error'" class="status error">
      服务连接失败，请检查后重新连接
      <button @click="checkApi">重新连接</button>
    </div>
    <div v-else class="status ok">
      服务已连接 
      <span v-if="currentModelName">| 当前模型: {{ currentModelName }}</span>
      <span v-else>| 请选择模型</span>
    </div>

    <!-- 模型选择 -->
    <div class="model-selector" v-if="models.length > 0">
      <label>选择模型:</label>
      <select v-model="selectedModel" @change="onModelChange" :disabled="switchingModel">
        <option value="" disabled>-- 选择模型 --</option>
        <option v-for="model in models" :key="model.id" :value="model.id" :disabled="!model.exists">
          {{ model.name }} {{ model.loaded ? '(已加载)' : model.exists ? '(未加载)' : '(不可用)' }}
        </option>
      </select>
      <button v-if="selectedModel && !isCurrentModelLoaded" @click="loadSelectedModel" :disabled="switchingModel" class="load-btn">
        {{ switchingModel ? '加载中...' : '加载模型' }}
      </button>
      <span v-else-if="isCurrentModelLoaded" class="loaded-text">✓ 已加载</span>
    </div>

    <div class="content">
      <!-- 输入 -->
      <div class="input-section">
        <h2>实体输入</h2>
        
        <!-- 参考用例 -->
        <div class="examples-section">
          <label>参考用例:</label>
          <select v-model="selectedExample" @change="applyExample">
            <option value="">-- 选择示例 --</option>
            <option v-for="(ex, idx) in currentExamples" :key="idx" :value="idx">
              {{ ex.name }}
            </option>
          </select>
        </div>

        <div class="entity-box">
          <h3>源实体 ({{ currentModelConfig?.left_range?.join('-') || '左图谱' }})</h3>
          <input v-model="source.id" placeholder="ID (如: 0)" type="number" />
          <input v-model="source.name" placeholder="名称" />
          <textarea v-model="source.text" placeholder="描述文本" rows="2"></textarea>
        </div>

        <div class="vs">VS</div>

        <div class="entity-box">
          <h3>目标实体 ({{ currentModelConfig?.right_range?.join('-') || '右图谱' }})</h3>
          <input v-model="target.id" placeholder="ID (如: 10500)" type="number" />
          <input v-model="target.name" placeholder="名称" />
          <textarea v-model="target.text" placeholder="描述文本" rows="2"></textarea>
        </div>

        <button class="align-btn" @click="runAlign" :disabled="loading">
          {{ loading ? '计算中...' : '执行对齐' }}
        </button>
      </div>

      <!-- 结果 -->
      <div class="result-section">
        <h2>对齐结果</h2>
        
        <div v-if="!result" class="placeholder">
          <p>输入两个实体并点击"执行对齐"</p>
          <div class="quick-tips">
            <h4>快速提示:</h4>
            <ul>
              <li>选择上方"参考用例"快速填充实体对</li>
              <li>源实体ID应在左图谱范围内</li>
              <li>目标实体ID应在右图谱范围内</li>
              <li>相似度 > 0.65 表示可对齐</li>
            </ul>
          </div>
        </div>

        <div v-else class="result">
          <div class="score-card">
            <span>相似度</span>
            <strong :style="{color: result.similarity > 0.65 ? '#4CAF50' : '#f44336'}">
              {{ (result.similarity * 100).toFixed(1) }}%
            </strong>
            <div class="bar"><div :style="{width: result.similarity * 100 + '%'}"></div></div>
            <span :class="['badge', result.is_aligned ? 'ok' : 'fail']">
              {{ result.is_aligned ? '✓ 可对齐' : '✗ 不可对齐' }}
            </span>
          </div>

          <div class="features" v-if="result.feature_scores">
            <h4>特征分解</h4>
            <div v-for="(score, name) in result.feature_scores" :key="name" class="feat-item">
              <span>{{ featureNames[name] || name }}</span>
              <div class="feat-bar"><div :style="{width: score * 100 + '%'}"></div></div>
              <span>{{ (score * 100).toFixed(0) }}%</span>
            </div>
          </div>

          <div class="attention" v-if="result.attention_map">
            <h4>融合权重</h4>
            <div class="attention-bars">
              <div v-for="(weight, idx) in result.attention_map" :key="idx" class="att-item">
                <span>{{ attentionLabels[idx] }}</span>
                <div class="att-bar"><div :style="{width: weight * 100 + '%'}"></div></div>
                <span>{{ (weight * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>

          <div class="result-details">
            <p>源实体ID: {{ result.source_id }}</p>
            <p>目标实体ID: {{ result.target_id }}</p>
            <p>欧氏距离: {{ result.distance }}</p>
          </div>

          <p class="info">耗时: {{ result.inference_time }}ms | {{ result.model_version }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { getModels, getModelInfo, switchModel, alignEntities } from '../api/index.js'

// ========== 参考用例 ==========
const EXAMPLES = {
  'zh_en': [
    { name: '示例1: 阿卜杜拉·居尔 ↔ Abdullah Gül', source: { id: 0, name: '阿卜杜拉·居尔', text: '土耳其第11任总统' }, target: { id: 10500, name: 'Abdullah Gül', text: '11th President of Turkey' } },
    { name: '示例2: 金大中 ↔ Kim Dae-jung', source: { id: 1, name: '金大中', text: '韩国第15任总统' }, target: { id: 10501, name: 'Kim Dae-jung', text: '8th President of South Korea' } },
    { name: '示例3: 恩斯特·金恩 ↔ Ernest King', source: { id: 5, name: '恩斯特·金恩', text: '美国海军五星上将' }, target: { id: 10505, name: 'Ernest King', text: 'United States Navy admiral' } },
    { name: '示例4: 约阿希姆·高克 ↔ Joachim Gauck', source: { id: 9, name: '约阿希姆·高克', text: '德国第11任联邦总统' }, target: { id: 10509, name: 'Joachim Gauck', text: 'President of Germany' } },
    { name: '示例5: 约翰·沃尔 ↔ John Wall', source: { id: 11, name: '约翰·沃尔', text: 'NBA篮球运动员' }, target: { id: 10511, name: 'John Wall', text: 'American basketball player' } },
    { name: '测试: 不匹配实体对', source: { id: 0, name: '阿卜杜拉·居尔', text: '土耳其总统' }, target: { id: 10511, name: 'John Wall', text: 'NBA球员' } },
  ],
  'FB15K_DB15K': [
    { name: '示例1: 自由实体对 (0 ↔ 15000)', source: { id: 0, name: 'Entity 0', text: 'FreeBase实体' }, target: { id: 15000, name: 'Entity 15000', text: 'DBPedia实体' } },
    { name: '示例2: 自由实体对 (10 ↔ 15010)', source: { id: 10, name: 'Entity 10', text: 'FreeBase实体' }, target: { id: 15010, name: 'Entity 15010', text: 'DBPedia实体' } },
    { name: '示例3: 自由实体对 (100 ↔ 15100)', source: { id: 100, name: 'Entity 100', text: 'FreeBase实体' }, target: { id: 15100, name: 'Entity 15100', text: 'DBPedia实体' } },
    { name: '示例4: 自由实体对 (500 ↔ 15500)', source: { id: 500, name: 'Entity 500', text: 'FreeBase实体' }, target: { id: 15500, name: 'Entity 15500', text: 'DBPedia实体' } },
    { name: '示例5: 自由实体对 (1000 ↔ 16000)', source: { id: 1000, name: 'Entity 1000', text: 'FreeBase实体' }, target: { id: 16000, name: 'Entity 16000', text: 'DBPedia实体' } },
  ],
  'FB15K_YAGO15K': [
    { name: '示例1: 自由实体对 (0 ↔ 15000)', source: { id: 0, name: 'Entity 0', text: 'FreeBase实体' }, target: { id: 15000, name: 'Entity 15000', text: 'YAGO实体' } },
    { name: '示例2: 自由实体对 (10 ↔ 15010)', source: { id: 10, name: 'Entity 10', text: 'FreeBase实体' }, target: { id: 15010, name: 'Entity 15010', text: 'YAGO实体' } },
    { name: '示例3: 自由实体对 (100 ↔ 15100)', source: { id: 100, name: 'Entity 100', text: 'FreeBase实体' }, target: { id: 15100, name: 'Entity 15100', text: 'YAGO实体' } },
    { name: '示例4: 自由实体对 (500 ↔ 15500)', source: { id: 500, name: 'Entity 500', text: 'FreeBase实体' }, target: { id: 15500, name: 'Entity 15500', text: 'YAGO实体' } },
    { name: '示例5: 自由实体对 (1000 ↔ 16000)', source: { id: 1000, name: 'Entity 1000', text: 'FreeBase实体' }, target: { id: 16000, name: 'Entity 16000', text: 'YAGO实体' } },
  ]
}

// ========== 状态 ==========
const apiStatus = ref('checking')
const models = ref([])
const selectedModel = ref('')
const currentModelName = ref('')
const switchingModel = ref(false)
const selectedExample = ref('')

const source = ref({ id: '', name: '', text: '' })
const target = ref({ id: '', name: '', text: '' })
const loading = ref(false)
const result = ref(null)

// ========== 计算属性 ==========
const currentExamples = computed(() => {
  if (!selectedModel.value) return []
  return EXAMPLES[selectedModel.value] || []
})

const currentModelConfig = computed(() => {
  if (!selectedModel.value) return null
  return models.value.find(m => m.id === selectedModel.value)
})

const featureNames = {
  structural: '结构',
  relational: '关系',
  attribute: '属性',
  visual: '视觉',
  name: '名称',
  character: '字符'
}

const attentionLabels = ['结构', '关系', '属性', '视觉', '名称', '字符']

const isCurrentModelLoaded = computed(() => {
  const model = models.value.find(m => m.id === selectedModel.value)
  return model?.loaded || false
})

// ========== 方法 ==========
const checkApi = async () => {
  apiStatus.value = 'checking'
  try {
    const res = await getModelInfo()
    if (res.success) {
      apiStatus.value = 'ok'
      await loadModels()
    } else {
      apiStatus.value = 'error'
    }
  } catch {
    apiStatus.value = 'error'
  }
}

const loadModels = async () => {
  try {
    const res = await getModels()
    if (res.success) {
      models.value = res.data
      // 设置当前模型
      const current = res.data.find(m => m.is_current && m.loaded)
      if (current) {
        selectedModel.value = current.id
        currentModelName.value = current.name
      } else if (res.data.length > 0) {
        // 默认选择第一个可用模型（仅设置选中，不自动加载）
        const firstAvailable = res.data.find(m => m.exists)
        if (firstAvailable) {
          selectedModel.value = firstAvailable.id
        }
      }
    }
  } catch (e) {
    console.error('加载模型列表失败:', e)
  }
}

const onModelChange = async () => {
  if (!selectedModel.value) {
    currentModelName.value = ''
    return
  }
  
  // 更新显示状态
  const modelInfo = models.value.find(m => m.id === selectedModel.value)
  if (modelInfo?.loaded) {
    currentModelName.value = modelInfo.name
  } else {
    currentModelName.value = ''
  }
}

const loadSelectedModel = async () => {
  if (!selectedModel.value) return
  
  const modelInfo = models.value.find(m => m.id === selectedModel.value)
  if (!modelInfo) return
  
  switchingModel.value = true
  try {
    const res = await switchModel(selectedModel.value)
    if (res.success) {
      currentModelName.value = modelInfo.name
      // 更新模型列表状态
      models.value = models.value.map(m => ({
        ...m,
        loaded: m.id === selectedModel.value,
        is_current: m.id === selectedModel.value
      }))
      // 清空结果和选择
      result.value = null
      selectedExample.value = ''
      source.value = { id: '', name: '', text: '' }
      target.value = { id: '', name: '', text: '' }
    } else {
      alert('加载模型失败: ' + (res.error || '未知错误'))
    }
  } catch (e) {
    alert('加载模型失败: ' + e.message)
  } finally {
    switchingModel.value = false
  }
}

const applyExample = () => {
  if (selectedExample.value === '') return
  const example = currentExamples.value[selectedExample.value]
  if (example) {
    source.value = { ...example.source }
    target.value = { ...example.target }
  }
}

const runAlign = async () => {
  if (source.value.id === '' || source.value.id === null || source.value.id === undefined ||
      target.value.id === '' || target.value.id === null || target.value.id === undefined) {
    alert('请输入源实体和目标实体的ID')
    return
  }
  
  // 检查模型是否已加载
  if (!isCurrentModelLoaded.value) {
    alert('请先选择并加载模型')
    return
  }
  
  loading.value = true
  result.value = null
  
  try {
    const res = await alignEntities(source.value, target.value)
    if (res.success) {
      result.value = res.data
    } else {
      alert('对齐失败: ' + (res.error || '未知错误'))
    }
  } catch (e) {
    alert('对齐失败: ' + e.message)
  } finally {
    loading.value = false
  }
}

onMounted(checkApi)
</script>

<style scoped>
.model-demo { padding: 30px; max-width: 1200px; margin: 0 auto; }
h1 { text-align: center; margin-bottom: 20px; }

/* 状态栏 */
.status { text-align: center; padding: 10px; margin-bottom: 15px; border-radius: 4px; }
.status.checking { background: #fff3e0; }
.status.error { background: #ffebee; color: #c62828; }
.status.ok { background: #e8f5e9; color: #2e7d32; }
.status button { margin-left: 10px; padding: 4px 12px; }

/* 模型选择 */
.model-selector {
  background: white;
  padding: 15px 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
}
.model-selector label { font-weight: bold; color: #555; }
.model-selector select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  min-width: 250px;
  font-size: 0.95rem;
}
.loading-text { color: #667eea; font-size: 0.9rem; }
.load-btn {
  padding: 8px 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
}
.load-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}
.loaded-text {
  color: #4CAF50;
  font-size: 0.9rem;
  font-weight: bold;
}

/* 主内容区 */
.content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
.input-section, .result-section { background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
h2 { margin-bottom: 20px; color: #667eea; font-size: 1.2rem; }

/* 参考用例 */
.examples-section {
  margin-bottom: 20px;
  padding: 15px;
  background: #f0f4ff;
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.examples-section label { font-weight: bold; color: #667eea; font-size: 0.9rem; }
.examples-section select {
  flex: 1;
  padding: 8px;
  border: 1px solid #c5cae9;
  border-radius: 4px;
  font-size: 0.9rem;
}

/* 实体输入 */
.entity-box { background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 15px; }
.entity-box h3 { margin-bottom: 10px; font-size: 0.95rem; color: #555; }
.entity-box input, .entity-box textarea { width: 100%; padding: 10px; margin-bottom: 8px; border: 1px solid #ddd; border-radius: 4px; }
.vs { text-align: center; font-weight: bold; color: #667eea; margin: 10px 0; }

.align-btn { width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }
.align-btn:disabled { opacity: 0.7; cursor: not-allowed; }

/* 结果区 */
.placeholder { text-align: center; padding: 40px 20px; color: #999; }
.quick-tips { margin-top: 30px; text-align: left; background: #f8f9fa; padding: 15px; border-radius: 6px; }
.quick-tips h4 { margin-bottom: 10px; color: #667eea; }
.quick-tips ul { margin-left: 20px; line-height: 1.8; }
.quick-tips li { color: #666; }

.score-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 20px; }
.score-card strong { display: block; font-size: 2.5rem; margin: 10px 0; }
.bar { height: 10px; background: #e0e0e0; border-radius: 5px; margin: 15px 0; overflow: hidden; }
.bar div { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.3s; }
.badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-size: 0.9rem; }
.badge.ok { background: #4CAF50; color: white; }
.badge.fail { background: #f44336; color: white; }

/* 特征分解 */
.features h4, .attention h4 { margin-bottom: 12px; color: #555; }
.feat-item { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.feat-item span:first-child { width: 60px; font-size: 0.9rem; }
.feat-bar { flex: 1; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; }
.feat-bar div { height: 100%; background: #667eea; transition: width 0.3s; }
.feat-item span:last-child { width: 45px; text-align: right; font-size: 0.85rem; }

/* 注意力权重 */
.attention { margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; }
.attention-bars { display: flex; flex-direction: column; gap: 8px; }
.att-item { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; }
.att-item span:first-child { width: 50px; color: #666; }
.att-bar { flex: 1; height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; }
.att-bar div { height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); }
.att-item span:last-child { width: 50px; text-align: right; color: #888; }

/* 结果详情 */
.result-details { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 6px; font-size: 0.85rem; color: #666; }
.result-details p { margin: 5px 0; }

.info { margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee; color: #888; font-size: 0.85rem; text-align: center; }

@media (max-width: 900px) {
  .content { grid-template-columns: 1fr; }
  .model-selector { flex-direction: column; align-items: stretch; }
  .examples-section { flex-direction: column; align-items: stretch; }
}
</style>
