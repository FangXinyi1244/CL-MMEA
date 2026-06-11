<template>
  <div class="app">
    <!-- 导航栏 -->
    <nav class="navbar">
      <div class="logo" @click="$router.push('/')">
        <span class="logo-icon">🔗</span>
        <span class="logo-text">MMEA</span>
      </div>
      <ul class="nav-links">
        <li><router-link to="/">首页</router-link></li>
        <li><router-link to="/graph">知识图谱</router-link></li>
        <li><router-link to="/demo">模型演示</router-link></li>
        <li><router-link to="/results">实验结果</router-link></li>
      </ul>
    </nav>
    
    <!-- 路由视图 -->
    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>

    <!-- 页脚 -->
    <footer class="footer">
      <p>多模态实体对齐研究演示系统 | 毕业设计项目</p>
    </footer>
  </div>
</template>

<script setup>
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 40px;
  height: 60px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
}

.logo-icon {
  font-size: 1.8rem;
}

.logo-text {
  font-size: 1.5rem;
  font-weight: bold;
  letter-spacing: 2px;
}

.nav-links {
  display: flex;
  gap: 5px;
  list-style: none;
}

.nav-links a {
  color: white;
  text-decoration: none;
  padding: 8px 20px;
  border-radius: 20px;
  transition: all 0.3s;
  font-size: 0.95rem;
}

.nav-links a:hover {
  background: rgba(255,255,255,0.2);
}

.nav-links a.router-link-active {
  background: rgba(255,255,255,0.25);
  font-weight: bold;
}

.main-content {
  flex: 1;
  background: #f5f7fa;
}

.footer {
  background: #2c3e50;
  color: #95a5a6;
  text-align: center;
  padding: 20px;
  font-size: 0.9rem;
}

/* 页面过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 滚动条样式 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

@media (max-width: 768px) {
  .navbar {
    padding: 0 20px;
  }
  .logo-text {
    font-size: 1.2rem;
  }
  .nav-links a {
    padding: 6px 12px;
    font-size: 0.85rem;
  }
}
</style>
