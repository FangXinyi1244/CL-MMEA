import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/graph', name: 'GraphView', component: () => import('../views/GraphView.vue') },
  { path: '/demo', name: 'ModelDemo', component: () => import('../views/ModelDemo.vue') },
  { path: '/results', name: 'ResultsView', component: () => import('../views/ResultsView.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
