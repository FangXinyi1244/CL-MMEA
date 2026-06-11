<template>
  <div class="home">
    <div class="hero">
      <h1>多模态实体对齐系统</h1>
      <p class="subtitle">Multi-Modal Entity Alignment Visualization</p>
      <p class="description">
        基于深度学习的多模态知识图谱实体对齐研究，支持图文多模态特征融合
      </p>
      <div class="features">
        <div class="feature-card" @click="$router.push('/graph')">
          <div class="icon">🔍</div>
          <h3>知识图谱可视化</h3>
          <p>交互式浏览多模态知识图谱结构</p>
        </div>
        <div class="feature-card" @click="$router.push('/demo')">
          <div class="icon">⚙️</div>
          <h3>模型演示</h3>
          <p>实时体验实体对齐模型推理</p>
        </div>
        <div class="feature-card" @click="$router.push('/results')">
          <div class="icon">📊</div>
          <h3>实验结果</h3>
          <p>查看模型训练效果对比分析</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
</script>

<style scoped>
.home {
  min-height: calc(100vh - 60px);
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
}

.hero {
  text-align: center;
  color: white;
  max-width: 1000px;
}

h1 {
  font-size: 3rem;
  margin-bottom: 10px;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.subtitle {
  font-size: 1.3rem;
  opacity: 0.9;
  margin-bottom: 20px;
}

.description {
  font-size: 1.1rem;
  opacity: 0.8;
  margin-bottom: 50px;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.features {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 30px;
  margin-top: 40px;
}

.feature-card {
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(10px);
  padding: 30px;
  border-radius: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 1px solid rgba(255,255,255,0.2);
}

.feature-card:hover {
  transform: translateY(-10px);
  background: rgba(255,255,255,0.25);
}

.icon {
  font-size: 3rem;
  margin-bottom: 15px;
}

.feature-card h3 {
  font-size: 1.3rem;
  margin-bottom: 10px;
}

.feature-card p {
  opacity: 0.9;
  font-size: 0.95rem;
}

@media (max-width: 768px) {
  .features {
    grid-template-columns: 1fr;
  }
  h1 {
    font-size: 2rem;
  }
}
</style>
