<template>
  <div class="results-view">
    <h1>📊 模型训练效果分析</h1>
    
    <!-- 方案总览 -->
    <section class="overview-section">
      <h2>方案综合对比</h2>
      <div class="comparison-grid">
        <div 
          v-for="(img, idx) in comparisonImages" 
          :key="idx"
          class="comparison-card"
          @click="openImage(img.src)"
        >
          <img :src="img.src" :alt="img.title" />
          <div class="card-info">
            <h3>{{ img.title }}</h3>
            <p>{{ img.desc }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- 方案详情 -->
    <section class="detail-section">
      <h2>各方案详细分析</h2>
      
      <div class="tabs">
        <button 
          v-for="tab in tabs" 
          :key="tab.id"
          :class="{ active: currentTab === tab.id }"
          @click="currentTab = tab.id"
        >
          {{ tab.name }}
        </button>
      </div>
      
      <div class="tab-content">
        <div class="detail-grid">
          <div class="chart-item">
            <h4>测试指标</h4>
            <img :src="getDetailImage('test_metrics_detailed.png')" @click="openModal($event)" />
          </div>
          <div class="chart-item">
            <h4>训练损失</h4>
            <img :src="getDetailImage('training_loss_detailed.png')" @click="openModal($event)" />
          </div>
          <div class="chart-item">
            <h4>权重分析</h4>
            <img :src="getDetailImage('weight_analysis_detailed.png')" @click="openModal($event)" />
          </div>
          <div class="chart-item">
            <h4>权重堆叠图</h4>
            <img :src="getDetailImage('weight_stackplot.png')" @click="openModal($event)" />
          </div>
        </div>
      </div>
    </section>

    <!-- 方案说明 -->
    <section class="scheme-section">
      <h2>方案说明</h2>
      <div class="scheme-cards">
        <div class="scheme-card">
          <h3>原方案 MCLEA</h3>
          <span class="tag baseline">基线</span>
          <p>多模态对比学习实体对齐 baseline 模型，基于视觉-文本双编码器结构。</p>
        </div>
        <div class="scheme-card">
          <h3>改进方案1</h3>
          <span class="tag improvement">+特征掩码</span>
          <p>在MCLEA基础上增加特征掩码机制，提升模型对缺失模态的鲁棒性。</p>
        </div>
        <div class="scheme-card">
          <h3>改进方案2</h3>
          <span class="tag improvement">+硬负采样</span>
          <p>在方案1基础上引入硬负采样策略，增强模型对困难样本的区分能力。</p>
        </div>
      </div>
    </section>

    <!-- 图片放大弹窗 -->
    <div v-if="modalImage" class="image-modal" @click="modalImage = null">
      <img :src="modalImage" @click.stop />
      <span class="close-modal">×</span>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const currentTab = ref('baseline')
const modalImage = ref(null)

const tabs = [
  { id: 'baseline', name: '原方案MCLEA（基线）', folder: '原方案MCLEA（基线）' },
  { id: 'improve1', name: '改进方案1（特征掩码）', folder: '改进方案1（只增加特征掩码）' },
  { id: 'improve2', name: '改进方案2（特征掩码+硬负采样）', folder: '改进方案2（特征掩码+硬负采样）' },
]

const comparisonImages = [
  { 
    title: '距离指标对比', 
    src: '/src/assets/data/distance_metrics_comparison.png',
    desc: '不同距离度量下的性能表现对比' 
  },
  { 
    title: '最终指标汇总', 
    src: '/src/assets/data/final_metrics_table.png',
    desc: '各方案最终评估指标汇总表' 
  },
  { 
    title: '测试指标对比', 
    src: '/src/assets/data/test_metrics_comparison.png',
    desc: '测试集上的各项性能指标对比' 
  },
  { 
    title: '训练损失曲线', 
    src: '/src/assets/data/training_loss_comparison.png',
    desc: '训练过程中的损失变化趋势' 
  },
  { 
    title: '权重演化对比', 
    src: '/src/assets/data/weight_evolution_comparison.png',
    desc: '多模态权重动态变化过程' 
  },
]

const getDetailImage = (filename) => {
  const folder = tabs.find(t => t.id === currentTab.value)?.folder
  return `/src/assets/data/${folder}/${filename}`
}

const openImage = (src) => {
  modalImage.value = src
}

const openModal = (event) => {
  modalImage.value = event.target.src
}
</script>

<style scoped>
.results-view {
  padding: 30px;
  max-width: 1400px;
  margin: 0 auto;
}

h1 {
  text-align: center;
  margin-bottom: 40px;
  color: #333;
}

h2 {
  margin-bottom: 20px;
  color: #667eea;
  font-size: 1.5rem;
  border-left: 4px solid #667eea;
  padding-left: 15px;
}

section {
  margin-bottom: 50px;
}

/* 总览区域 */
.comparison-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}

.comparison-card {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  cursor: pointer;
  transition: transform 0.3s, box-shadow 0.3s;
}

.comparison-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.comparison-card img {
  width: 100%;
  height: 200px;
  object-fit: cover;
}

.card-info {
  padding: 20px;
}

.card-info h3 {
  margin-bottom: 8px;
  color: #333;
}

.card-info p {
  color: #666;
  font-size: 0.9rem;
}

/* 详情区域 */
.detail-section {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 25px;
  flex-wrap: wrap;
}

.tabs button {
  padding: 12px 24px;
  border: 2px solid #e0e0e0;
  background: white;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
  font-size: 0.95rem;
}

.tabs button:hover {
  border-color: #667eea;
  color: #667eea;
}

.tabs button.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 25px;
}

.chart-item {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
}

.chart-item h4 {
  margin-bottom: 15px;
  color: #555;
  text-align: center;
}

.chart-item img {
  width: 100%;
  border-radius: 8px;
  cursor: zoom-in;
  transition: transform 0.3s;
}

.chart-item img:hover {
  transform: scale(1.02);
}

/* 方案说明 */
.scheme-section {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.scheme-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.scheme-card {
  padding: 25px;
  background: #f8f9fa;
  border-radius: 8px;
  position: relative;
}

.scheme-card h3 {
  margin-bottom: 10px;
  color: #333;
}

.scheme-card p {
  color: #666;
  font-size: 0.95rem;
  line-height: 1.6;
}

.tag {
  display: inline-block;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  margin-bottom: 15px;
}

.tag.baseline {
  background: #e3f2fd;
  color: #1976d2;
}

.tag.improvement {
  background: #e8f5e9;
  color: #388e3c;
}

/* 图片弹窗 */
.image-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  cursor: zoom-out;
}

.image-modal img {
  max-width: 90%;
  max-height: 90%;
  border-radius: 8px;
}

.close-modal {
  position: absolute;
  top: 20px;
  right: 30px;
  font-size: 3rem;
  color: white;
  cursor: pointer;
}

@media (max-width: 768px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }
  .scheme-cards {
    grid-template-columns: 1fr;
  }
  .tabs button {
    font-size: 0.85rem;
    padding: 10px 15px;
  }
}
</style>
